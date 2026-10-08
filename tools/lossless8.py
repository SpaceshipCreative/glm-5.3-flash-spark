#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 SpaceshipCreative and glm-5.3-flash-spark contributors.
# Attribution: the two-candidate MXFP8 exponent rule and the block-128 "amax / 448" FP8 grid are adapted
# from knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4 scripts/glm_quant_mix.py (q_mxfp8, q_fp8_blk) at commit
# 770d115, MIT License, Copyright (c) 2026 knapcio. https://github.com/knapcio/GLM-5.3-Flash-4x-DGX-Spark-TP4
"""lossless8: re-encode the BF16 projections of nvidia/GLM-5.3-Flash-NVFP4 on their native 8-bit grids.

Z.ai's BF16 weights came out of 8-bit training grids. The MLA, shared-expert and indexer projections sit on a
128x128 block-FP8 grid (scale = block amax / 448, which gives the same codes as Z.ai's FP8 release). The KDA
projections sit on an MXFP8 grid (1x32 blocks with an E8M0 scale). Finding that grid again halves the bytes,
and about 90% of elements decode bit-exactly. A group is converted only if every member passes the gate
(max relative Frobenius error and min bit-exact fraction). Everything else stays BF16.

The output is a ModelOpt MIXED_PRECISION checkpoint that vLLM v0.31.0 loads without loader changes:
  FP8_PB_WO  weight F8_E4M3 [N,K] + weight_scale F32 [ceil(N/128),1,K/128,1]
  MXFP8      weight F8_E4M3 [N,K] + weight_scale U8 (E8M0) [N,K/32]
NVFP4 layers (routed experts and dense MLP) are listed unchanged. Serving it needs the p5-lossless8 vLLM
patches, which let the GLM KDA/MLA projections honour a MIXED_PRECISION config. spark.sh routes
fp8_block_w8a8 and mxfp8 to Marlin (W8A16) through --kernel-config linear_backend_per_quant.

  lossless8.py SRC DST [--dry-run] [--include RE] [--exclude RE] [--tp 2,4] [--max-rel-err 0.005]
  lossless8.py --drafter SRC DST     # DFlash2 drafter: lossy block FP8, no gate (cf. knapcio drafter_fp8.py)
  lossless8.py --self-test           # CPU only
"""
import argparse
import fnmatch
import json
import os
import re
import shutil
import struct
import sys
import tempfile

FP8_MAX, BLK, MX = 448.0, 128, 32
ALGO = {"fp8blk": "FP8_PB_WO", "mxfp8": "MXFP8"}
CONFIGS = ("config.json", "hf_quant_config.json", "model.safetensors.index.json")

# kind -> (layer type, vLLM module, [(checkpoint member, TP split)], formats in preference order)
# split: col = output rows sharded over TP, row = input dim sharded, rep = replicated on every rank.
_GRID_BLK, _GRID_MX = ("fp8blk", "mxfp8"), ("mxfp8", "fp8blk")
KINDS = {
    "kda_in": ("kda", "self_attn.in_proj_qkvbfg_a",
               [("self_attn.q_proj", "col"), ("self_attn.k_proj", "col"), ("self_attn.v_proj", "col"),
                ("self_attn.b_proj", "col"), ("self_attn.f_a_proj", "rep"), ("self_attn.g_a_proj", "rep")], _GRID_MX),
    "kda_fb": ("kda", "self_attn.f_b_proj", [("self_attn.f_b_proj", "col")], _GRID_MX),
    "kda_gb": ("kda", "self_attn.g_b_proj", [("self_attn.g_b_proj", "col")], _GRID_MX),
    "kda_o": ("kda", "self_attn.o_proj", [("self_attn.o_proj", "row")], _GRID_MX),
    "mla_qa": ("mla", "self_attn.fused_qkv_a_proj",
               [("self_attn.q_a_proj", "rep"), ("self_attn.kv_a_proj_with_mqa", "rep")], _GRID_BLK),
    "mla_qb": ("mla", "self_attn.q_b_proj", [("self_attn.q_b_proj", "col")], _GRID_BLK),
    "mla_o": ("mla", "self_attn.o_proj", [("self_attn.o_proj", "row")], _GRID_BLK),
    "idx_qb": ("mla", "self_attn.indexer.wq_b", [("self_attn.indexer.wq_b", "rep")], _GRID_BLK),
    "sh_gu": (None, "mlp.shared_experts.gate_up_proj",
              [("mlp.shared_experts.gate_proj", "col"), ("mlp.shared_experts.up_proj", "col")], _GRID_BLK),
    "sh_down": (None, "mlp.shared_experts.down_proj", [("mlp.shared_experts.down_proj", "row")], _GRID_BLK),
}
# Never converted: kv_b_proj (absorbed into BF16 W_UK/W_UV), indexer wk/weights_proj (fused BF16), router gate,
# lm_head, embeddings, mHC hc_*, norms, conv1d, visual, MTP layers.
DRAFTER_KINDS = {
    "qkv": (None, "self_attn.qkv_proj",
            [("self_attn.q_proj", "col"), ("self_attn.k_proj", "col"), ("self_attn.v_proj", "col")], ("fp8blk",)),
    "o": (None, "self_attn.o_proj", [("self_attn.o_proj", "row")], ("fp8blk",)),
    "gate_up": (None, "mlp.gate_up_proj", [("mlp.gate_proj", "col"), ("mlp.up_proj", "col")], ("fp8blk",)),
    "down": (None, "mlp.down_proj", [("mlp.down_proj", "row")], ("fp8blk",)),
}
DRAFTER_IGNORE = ["lm_head", "*lm_head", "*embed_tokens", "*fc", "*kernel_projection", "*hidden_projection"]
# Replicated on every TP rank at decode (read in full); everything else non-expert is read 1/tp per rank.
REPLICATED = re.compile(r"q_a_proj|kv_a_proj_with_mqa|\.indexer\.|\.mlp\.gate\.|[fg]_a_proj|\.hc_|norm")


# ---------------------------------------------------------------- quantizers (torch imported lazily)
def q_fp8blk(w):
    """128x128 block FP8: scale = block amax / 448 (adapted from knapcio q_fp8_blk). -> (F8 [N,K], F32 4-D)."""
    import torch
    n, k = w.shape
    ob, ib = -(-n // BLK), k // BLK
    t = torch.zeros(ob * BLK, k, dtype=torch.float32)
    t[:n] = w.float()
    t = t.view(ob, BLK, ib, BLK)
    s = t.abs().amax(dim=(1, 3)).clamp(min=1e-12) / FP8_MAX
    q = (t / s[:, None, :, None]).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    return q.view(ob * BLK, k)[:n].contiguous(), s.view(ob, 1, ib, 1).contiguous()


def dq_fp8blk(q, s):
    n = q.shape[0]
    s = s.reshape(s.shape[0], s.shape[2])
    return q.float() * s.repeat_interleave(BLK, 0)[:n].repeat_interleave(BLK, 1)


def q_mxfp8(w):
    """MXFP8 (1x32, E8M0). Per block, keep the lower-error exponent of ceil(log2(amax/448)) (no clipping) and the
    OCP rule floor(log2(amax)) - 8, which reproduces weights already on an MX grid (adapted from knapcio q_mxfp8)."""
    import torch
    n, k = w.shape
    b = w.float().view(n, k // MX, MX)
    amax = b.abs().amax(-1).clamp(min=2.0 ** -126)
    best = None
    for e in (torch.ceil(torch.log2(amax / FP8_MAX)), torch.floor(torch.log2(amax)) - 8):
        e = e.clamp(-127, 127)
        q = (b / torch.exp2(e).unsqueeze(-1)).clamp(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn).view(torch.uint8)
        err = ((q.view(torch.float8_e4m3fn).float() * torch.exp2(e).unsqueeze(-1) - b) ** 2).sum(-1)
        if best is None:
            best = [err, q, e]
        else:
            t = err < best[0]
            best = [torch.where(t, err, best[0]), torch.where(t.unsqueeze(-1), q, best[1]), torch.where(t, e, best[2])]
    return best[1].view(torch.float8_e4m3fn).view(n, k).contiguous(), (best[2] + 127).to(torch.uint8).contiguous()


def dq_mxfp8(q, s):
    import torch
    n, k = q.shape
    return (q.float().view(n, k // MX, MX) * torch.exp2(s.float() - 127).unsqueeze(-1)).view(n, k)


QUANT = {"fp8blk": (q_fp8blk, dq_fp8blk), "mxfp8": (q_mxfp8, dq_mxfp8)}


def scale_bytes(fmt, n, k):
    return 4 * -(-n // BLK) * (k // BLK) if fmt == "fp8blk" else n * (k // MX)


def stats(w, dq):
    import torch
    w = w.float()
    d = dq - w
    return {"rel": float(d.norm() / w.norm().clamp(min=1e-30)), "max_abs": float(d.abs().max()),
            "exact": float((dq.to(torch.bfloat16) == w.to(torch.bfloat16)).float().mean())}


def aligned(fmt, shape, split, tp):
    """Would every TP rank get whole blocks? FP8_PB_WO needs per-rank N and K % 128, MXFP8 per-rank K % 32."""
    n, k = shape
    if (split == "col" and n % tp) or (split == "row" and k % tp):
        return False
    rows, cols = (n // tp if split == "col" else n), (k // tp if split == "row" else k)
    return cols % MX == 0 if fmt == "mxfp8" else rows % BLK == 0 and cols % BLK == 0


# ---------------------------------------------------------------- checkpoint plumbing
def read_header(path):
    with open(path, "rb") as f:
        h = json.loads(f.read(struct.unpack("<Q", f.read(8))[0]))
    h.pop("__metadata__", None)
    return h


def load_index(src):
    """-> (shards, {tensor: (shard, dtype, shape, nbytes)}), from the safetensors headers alone."""
    idx = os.path.join(src, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            shards = sorted(set(json.load(f)["weight_map"].values()))
    else:
        shards = sorted(f for f in os.listdir(src) if f.endswith(".safetensors"))
    tensors = {}
    for s in shards:
        for k, v in read_header(os.path.join(src, s)).items():
            tensors[k] = (s, v["dtype"], v["shape"], v["data_offsets"][1] - v["data_offsets"][0])
    return shards, tensors


def plan(tensors, cfg, kinds, tps, include=None, exclude=None):
    tc = cfg.get("text_config", cfg)
    n_layers, layer_types = tc.get("num_hidden_layers"), tc.get("layer_types") or []
    prefixes = sorted({m.group(0) for n in tensors for m in [re.match(r"^.*?layers\.(\d+)\.", n)] if m},
                      key=lambda p: (int(p.rsplit(".", 2)[-2]), p))
    groups = []
    for pre in prefixes:
        i = int(pre.rsplit(".", 2)[-2])
        if n_layers is not None and i >= n_layers:
            continue  # MTP / nextn layers stay as shipped
        is_kda = i < len(layer_types) and layer_types[i] == "linear_attention"
        for kind, (ltype, module, members, fmts) in kinds.items():
            if (ltype == "kda" and not is_kda) or (ltype == "mla" and is_kda):
                continue
            names = [pre + m + ".weight" for m, _ in members]
            if not all(n in tensors and tensors[n][1] == "BF16" and len(tensors[n][2]) == 2 for n in names):
                continue
            key = " ".join([kind, pre + module] + names)
            if (include and not re.search(include, key)) or (exclude and re.search(exclude, key)):
                continue
            shapes = [tensors[n][2] for n in names]
            ok = [f for f in fmts if all(aligned(f, s, sp, tp) for s, (_, sp) in zip(shapes, members) for tp in tps)]
            groups.append({"kind": kind, "module": pre + module, "members": names, "shapes": shapes, "fmts": ok})
    return groups


def quantize_groups(groups, src, tensors, max_rel, min_exact, gated=True):
    """Quantize every group in each allowed format, keep the lowest-error one, apply the gate. -> {name: (q, s)}."""
    from safetensors import safe_open
    handles, out = {}, {}

    def get(name):
        s = tensors[name][0]
        if s not in handles:
            handles[s] = safe_open(os.path.join(src, s), "pt")
        return handles[s].get_tensor(name)

    for g in groups:
        if not g["fmts"]:
            g["status"] = "kept: no format divides evenly over --tp"
            continue
        ws, best = [get(n) for n in g["members"]], None
        for fmt in g["fmts"]:
            res = [QUANT[fmt][0](w) for w in ws]
            st = [stats(w, QUANT[fmt][1](*r)) for w, r in zip(ws, res)]
            worst = max(s["rel"] for s in st)
            if best is None or worst < best[0]:
                best = (worst, fmt, res, st)
        worst, g["fmt"], res, g["stats"] = best
        ok = not gated or (worst <= max_rel and min(s["exact"] for s in g["stats"]) >= min_exact)
        g["status"] = "converted" if ok else "kept: above gate"
        if ok:
            out.update(zip(g["members"], res))
        print(f"  {g['status']:<18} {g['fmt']:<7} rel {worst:.2e}  {g['module']}", flush=True)
    return out


def bitwise_equal(a, b):
    import torch
    return a.dtype == b.dtype and a.shape == b.shape and torch.equal(
        a.contiguous().reshape(-1).view(torch.uint8), b.contiguous().reshape(-1).view(torch.uint8))


def place(mode, a, b):
    a = os.path.realpath(a)
    if mode == "copy":
        shutil.copy2(a, b)
        return
    if mode == "hard":
        try:
            os.link(a, b)
            return
        except OSError:
            pass  # cross-device: fall back to a symlink
    os.symlink(a, b)


def write_checkpoint(src, dst, shards, conv, link):
    """Rewrite only the shards holding converted tensors (read back and compare bitwise); link the rest."""
    from safetensors import safe_open
    from safetensors.torch import save_file
    os.makedirs(dst, exist_ok=True)
    if os.listdir(dst):
        sys.exit(f"{dst} is not empty")
    affected = set()
    for f in os.listdir(src):
        p = os.path.join(src, f)
        if f not in CONFIGS and os.path.isfile(p) and f not in shards:
            place(link, p, os.path.join(dst, f))
    for s in shards:
        with safe_open(os.path.join(src, s), "pt") as f:
            names = list(f.keys())
            if not any(n in conv for n in names):
                place(link, os.path.join(src, s), os.path.join(dst, s))
                continue
            affected.add(s)
            out = {}
            for n in names:
                if n in conv:
                    out[n], out[n + "_scale"] = conv[n]
                else:
                    out[n] = f.get_tensor(n)
            save_file(out, os.path.join(dst, s), metadata=f.metadata())
            del out
        with safe_open(os.path.join(src, s), "pt") as a, safe_open(os.path.join(dst, s), "pt") as b:
            for n in names:
                want = conv[n] if n in conv else (a.get_tensor(n),)
                got = (b.get_tensor(n), b.get_tensor(n + "_scale")) if n in conv else (b.get_tensor(n),)
                assert all(bitwise_equal(x, y) for x, y in zip(want, got)), f"read-back mismatch: {s}:{n}"
        print(f"  rewrote {s}", flush=True)
    return affected


def write_index(src, dst, shards):
    p = os.path.join(src, "model.safetensors.index.json")
    if not os.path.exists(p):
        return
    with open(p) as f:
        idx = json.load(f)
    wm, total = {}, 0
    for s in shards:
        for k, v in read_header(os.path.join(dst, s)).items():
            wm[k] = s
            total += v["data_offsets"][1] - v["data_offsets"][0]
    idx["weight_map"] = dict(sorted(wm.items()))
    idx.setdefault("metadata", {})["total_size"] = total
    dump(idx, os.path.join(dst, "model.safetensors.index.json"))


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
        f.write("\n")


def _swap(name):
    a, b = "model.language_model.", "language_model.model."
    return b + name[len(a):] if name.startswith(a) else a + name[len(b):] if name.startswith(b) else name


def excluded(module, patterns):
    """vLLM ModelOptQuantConfigBase.is_layer_excluded: exact, substring or fnmatch, under either name mapping."""
    pats = set(patterns) | {_swap(p) for p in patterns}
    return any(m == p or p in m or fnmatch.fnmatch(m, p) for m in (module, _swap(module)) for p in pats)


def mixed_quant_config(cfg, hfq, tensors, groups):
    """-> (config.json quantization_config, hf_quant_config.json) for ModelOpt MIXED_PRECISION."""
    q, hq = cfg.get("quantization_config") or {}, (hfq or {}).get("quantization") or {}
    algo = str(hq.get("quant_algo") or q.get("quant_algo") or "").upper()
    if algo not in ("NVFP4", "W4A16_NVFP4"):
        sys.exit(f"expected an NVFP4 ModelOpt source checkpoint, got quant_algo={algo!r}")
    gs = int(hq.get("group_size") or q.get("group_size") or 16)
    ql = {}
    for name, (_, dtype, _, _) in tensors.items():
        if name.endswith(".weight_scale") and dtype == "F8_E4M3":  # NVFP4 per-16 block scale
            ql[re.sub(r"\.experts\.\d+\..*$", ".experts", name[: -len(".weight_scale")])] = {
                "quant_algo": algo, "group_size": gs}
    conv_mods = []
    for g in groups:
        if g["status"] == "converted":
            for m in [g["module"]] + [n[: -len(".weight")] for n in g["members"]]:
                ql[m] = {"quant_algo": ALGO[g["fmt"]]}
                conv_mods.append(m)
    ignore = [p for p in (q.get("ignore") or hq.get("exclude_modules") or [])
              if not any(excluded(m, [p]) for m in conv_mods)]
    assert not any(excluded(m, ignore) for m in conv_mods)
    ql = dict(sorted(ql.items()))
    producer = q.get("producer") or (hfq or {}).get("producer")
    qc = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "group_size": gs, "ignore": ignore,
          "quantized_layers": ql, "producer": producer}
    if q.get("kv_cache_scheme"):
        qc["kv_cache_scheme"] = q["kv_cache_scheme"]
    hf = {"producer": producer, "quantization": {
        "quant_algo": "MIXED_PRECISION", "kv_cache_quant_algo": hq.get("kv_cache_quant_algo"), "group_size": gs,
        "exclude_modules": ignore, "quantized_layers": ql}}
    return qc, hf


def drafter_quant_config(groups):
    dropped = sorted({"*." + g["module"].split(".", 2)[-1] for g in groups if g["status"] != "converted"})
    return {"quant_method": "modelopt", "quant_algo": "FP8_PB_WO", "ignore": DRAFTER_IGNORE + dropped}


# ---------------------------------------------------------------- byte accounting
def decode_bytes(tensors, cfg, tp, conv_fmt):
    """Weight bytes one TP rank reads per decoded token at batch 1 (experts: top_k/n_routed of the experts).
    Skips embeddings, visual and MTP layers. conv_fmt: {tensor: fmt} for converted tensors."""
    tc = cfg.get("text_config", cfg)
    n_layers = tc.get("num_hidden_layers")
    act = tc.get("num_experts_per_tok", 1) / max(tc.get("n_routed_experts") or 1, 1)
    total = 0.0
    for name, (_, dtype, shape, nb) in tensors.items():
        m = re.search(r"layers\.(\d+)\.", name)
        if "embed" in name or "visual" in name or (m and n_layers is not None and int(m.group(1)) >= n_layers):
            continue
        if name in conv_fmt:
            nb = shape[0] * shape[1] + scale_bytes(conv_fmt[name], *shape)
        frac = act / tp if ".experts." in name else 1.0 if REPLICATED.search(name) else 1.0 / tp
        total += nb * frac
    return total


def gib(x):
    return f"{x / 2 ** 30:7.2f} GiB"


def summarize(groups, tensors, cfg, tps, conv_fmt):
    kinds = {}
    for g in groups:
        k = kinds.setdefault(g["kind"], {"n": 0, "conv": 0, "fmt": set(), "bf16": 0, "new": 0, "worst": 0.0})
        k["n"] += 1
        for n, s in zip(g["members"], g["shapes"]):
            k["bf16"] += tensors[n][3]
            k["new"] += s[0] * s[1] + scale_bytes(conv_fmt[n], *s) if n in conv_fmt else tensors[n][3]
        k["worst"] = max([k["worst"]] + [s["rel"] for s in g.get("stats", [])])
        if n in conv_fmt:
            k["conv"] += 1
            k["fmt"].add(conv_fmt[n])
    print(f"{'kind':<9}{'groups':>7}{'conv':>6}  {'format':<13}{'BF16':>12}{'after':>12}  worst rel")
    for kind, k in kinds.items():
        print(f"{kind:<9}{k['n']:>7}{k['conv']:>6}  {','.join(sorted(k['fmt'])) or '-':<13}"
              f"{gib(k['bf16'])}{gib(k['new'])}  {k['worst']:.2e}")
    before = sum(t[3] for t in tensors.values())
    after = before - sum(tensors[n][3] - s[0] * s[1] - scale_bytes(f, *s)
                         for n, f in conv_fmt.items() for s in [tensors[n][2]])
    print(f"checkpoint: {gib(before)} -> {gib(after)}")
    for tp in tps:
        b, a = decode_bytes(tensors, cfg, tp, {}), decode_bytes(tensors, cfg, tp, conv_fmt)
        print(f"decode weight bytes/token/rank TP={tp}: {gib(b)} -> {gib(a)}  ({100 * (1 - a / b):.1f}% fewer)")


# ---------------------------------------------------------------- main
def convert(src, dst, *, drafter=False, dry_run=False, include=None, exclude=None, tps=(2, 4),
            max_rel=0.005, min_exact=0.5, link="hard"):
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    hfq_path = os.path.join(src, "hf_quant_config.json")
    hfq = json.load(open(hfq_path)) if os.path.exists(hfq_path) else None
    shards, tensors = load_index(src)
    groups = plan(tensors, cfg, DRAFTER_KINDS if drafter else KINDS, tps, include, exclude)
    print(f"{len(groups)} candidate groups in {len(shards)} shards")
    if dry_run:  # headers only: assume every group lands on its preferred format
        for g in groups:
            g["status"], g["fmt"] = ("converted", g["fmts"][0]) if g["fmts"] else ("kept: no format fits", None)
        conv_fmt = {n: g["fmt"] for g in groups if g["status"] == "converted" for n in g["members"]}
        touched = {tensors[n][0] for n in conv_fmt}
        print(f"plan: rewrite {len(touched)} shards, link {len(shards) - len(touched)}"
              " (dry run assumes every group passes the gate)")
        summarize(groups, tensors, cfg, tps, conv_fmt)
        return groups
    conv = quantize_groups(groups, src, tensors, max_rel, min_exact, gated=not drafter)
    conv_fmt = {n: g["fmt"] for g in groups if g["status"] == "converted" for n in g["members"]}
    write_checkpoint(src, dst, shards, conv, link)
    write_index(src, dst, shards)
    if drafter:
        cfg["quantization_config"] = drafter_quant_config(groups)
    else:
        cfg["quantization_config"], hf = mixed_quant_config(cfg, hfq, tensors, groups)
        if hfq is not None:
            dump(hf, os.path.join(dst, "hf_quant_config.json"))
    dump(cfg, os.path.join(dst, "config.json"))
    report = [{k: g.get(k) for k in ("kind", "module", "status", "fmt")} |
              {"members": [dict(name=n, **s) for n, s in zip(g["members"], g.get("stats", []))]} for g in groups]
    dump({"max_rel_err": max_rel, "min_exact": min_exact, "tp": list(tps), "groups": report},
         os.path.join(dst, "lossless8-report.json"))
    summarize(groups, tensors, cfg, tps, conv_fmt)
    return groups


def self_test():
    """CPU self-test: on-grid tensors round-trip bit-exactly, near-grid pass the gate, off-grid are rejected,
    and a two-layer fake checkpoint converts end to end."""
    import torch
    from safetensors import safe_open
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(0)
    codes = torch.arange(256, dtype=torch.int32).to(torch.uint8).view(torch.float8_e4m3fn)
    finite = codes[torch.isfinite(codes.float())]

    def rand_codes(n, k):  # random finite e4m3 codes
        return finite[torch.randint(len(finite), (n, k), generator=g)]

    def on_mx_grid(n, k):
        c = rand_codes(n, k).view(torch.uint8).view(n, k // MX, MX)
        c[..., 0] = 0x7E  # 448: block amax pins the exponent
        e = torch.randint(-20, -5, (n, k // MX), generator=g).float()
        w = (c.view(torch.float8_e4m3fn).float() * torch.exp2(e).unsqueeze(-1)).view(n, k).to(torch.bfloat16)
        return w, c.view(n, k), (e + 127).to(torch.uint8)

    def on_blk_grid(n, k):
        c = rand_codes(n, k).view(torch.uint8)
        for i in range(0, n, BLK):
            for j in range(0, k, BLK):
                c[i, j] = 0xFE  # -448
        mant = torch.randint(8, 16, (-(-n // BLK), k // BLK), generator=g).float()  # 4 significant bits
        s = mant * torch.exp2(torch.randint(-16, -10, mant.shape, generator=g).float())
        w = c.view(torch.float8_e4m3fn).float() * s.repeat_interleave(BLK, 0)[:n].repeat_interleave(BLK, 1)
        return w.to(torch.bfloat16), c, s.view(-1, 1, k // BLK, 1)

    # 1. MXFP8 on-grid: codes, scales and values bit-exact.
    w, c, e = on_mx_grid(64, 256)
    q, s = q_mxfp8(w)
    assert torch.equal(q.view(torch.uint8), c) and torch.equal(s, e)
    assert torch.equal(dq_mxfp8(q, s), w.float())
    # 2. Block FP8 on-grid, including a partial trailing block (n=200).
    for n in (256, 200):
        w, c, sc = on_blk_grid(n, 384)
        q, s = q_fp8blk(w)
        assert torch.equal(q.view(torch.uint8), c) and torch.equal(s, sc), n
        assert torch.equal(dq_fp8blk(q, s), w.float()), n
    # 3. Near-grid: 10% of the non-max, nonzero elements moved by one bf16 ulp keep their codes and pass the gate.
    w, c, sc = on_blk_grid(256, 256)
    bits = w.view(torch.int16).clone()
    hit = (torch.rand(w.shape, generator=g) < 0.1) & (w != 0) & (c & 0x7F != 0x7E)
    bits[hit] += 1
    w2 = bits.view(torch.bfloat16)
    q, s = q_fp8blk(w2)
    st = stats(w2, dq_fp8blk(q, s))
    assert torch.equal(q.view(torch.uint8), c) and st["rel"] < 0.005 and 0.85 < st["exact"] < 0.95, st
    # 4. Off-grid Gaussian weights fail the gate in both formats.
    w = (torch.randn(256, 256, generator=g) * 0.02).to(torch.bfloat16)
    for fmt, (qf, dqf) in QUANT.items():
        st = stats(w, dqf(*qf(w)))
        assert st["rel"] > 0.005 and st["exact"] < 0.5, (fmt, st)
    # 5. Fake two-layer checkpoint (KDA layer 0, MLA layer 1, MTP layer 2) through convert().
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "src"), os.path.join(tmp, "dst")
        os.makedirs(src)
        p0, p1, p2 = (f"model.language_model.layers.{i}." for i in range(3))
        a = {p0 + m + ".weight": on_mx_grid(*sh)[0] for m, sh in [
            ("self_attn.q_proj", (256, 256)), ("self_attn.k_proj", (256, 256)), ("self_attn.v_proj", (256, 256)),
            ("self_attn.b_proj", (4, 256)), ("self_attn.f_a_proj", (32, 256)), ("self_attn.g_a_proj", (32, 256)),
            ("self_attn.f_b_proj", (256, 32)), ("self_attn.g_b_proj", (256, 32)), ("self_attn.o_proj", (256, 256))]}
        a.update({p1 + m + ".weight": on_blk_grid(*sh)[0] for m, sh in [
            ("self_attn.q_a_proj", (256, 256)), ("self_attn.kv_a_proj_with_mqa", (128, 256)),
            ("self_attn.q_b_proj", (512, 256)), ("self_attn.o_proj", (256, 512)),
            ("mlp.shared_experts.gate_proj", (256, 256)), ("mlp.shared_experts.up_proj", (256, 256)),
            ("mlp.shared_experts.down_proj", (256, 256))]})
        a[p1 + "self_attn.indexer.wq_b.weight"] = (torch.randn(256, 256, generator=g) * 0.02).to(torch.bfloat16)
        a[p1 + "self_attn.kv_b_proj.weight"] = on_blk_grid(256, 128)[0]
        a[p2 + "self_attn.q_a_proj.weight"] = on_blk_grid(256, 256)[0]
        b = {"lm_head.weight": torch.randn(64, 256, generator=g).to(torch.bfloat16),
             p1 + "mlp.experts.0.down_proj.weight": torch.zeros(256, 64, dtype=torch.uint8),
             p1 + "mlp.experts.0.down_proj.weight_scale": torch.ones(256, 8).to(torch.float8_e4m3fn),
             p1 + "mlp.experts.0.down_proj.weight_scale_2": torch.ones(()),
             p1 + "mlp.experts.0.down_proj.input_scale": torch.ones(())}
        save_file(a, os.path.join(src, "model-00001-of-00002.safetensors"), metadata={"format": "pt"})
        save_file(b, os.path.join(src, "model-00002-of-00002.safetensors"), metadata={"format": "pt"})
        dump({"weight_map": {**{k: "model-00001-of-00002.safetensors" for k in a},
                             **{k: "model-00002-of-00002.safetensors" for k in b}}},
             os.path.join(src, "model.safetensors.index.json"))
        excl = ["lm_head", p0.rstrip(".") + ".self_attn*", p1 + "self_attn*", p1 + "mlp.shared_experts*",
                p1 + "mlp.gate", "model.language_model.layers.2*"]
        dump({"text_config": {"num_hidden_layers": 2, "layer_types": ["linear_attention", "deepseek_sparse_attention"],
                              "num_experts_per_tok": 1, "n_routed_experts": 1},
              "quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4", "ignore": excl,
                                      "kv_cache_scheme": {"type": "float", "num_bits": 8, "dynamic": False},
                                      "producer": {"name": "modelopt"}, "config_groups": {}}},
             os.path.join(src, "config.json"))
        dump({"producer": {"name": "modelopt"}, "quantization": {
            "quant_algo": "NVFP4", "kv_cache_quant_algo": "FP8", "group_size": 16, "exclude_modules": excl}},
            os.path.join(src, "hf_quant_config.json"))
        with open(os.path.join(src, "tokenizer.json"), "w") as f:
            f.write("{}")
        groups = convert(src, dst, tps=(2,))
        st = {g["kind"]: (g["status"], g.get("fmt")) for g in groups}
        assert st == {"kda_in": ("converted", "mxfp8"), "kda_fb": ("converted", "mxfp8"),
                      "kda_gb": ("converted", "mxfp8"), "kda_o": ("converted", "mxfp8"),
                      "mla_qa": ("converted", "fp8blk"), "mla_qb": ("converted", "fp8blk"),
                      "mla_o": ("converted", "fp8blk"), "idx_qb": ("kept: above gate", "fp8blk"),
                      "sh_gu": ("converted", "fp8blk"), "sh_down": ("converted", "fp8blk")}, st
        s1, s2 = (os.path.join(d, "model-00002-of-00002.safetensors") for d in (src, dst))
        assert os.stat(s1).st_ino == os.stat(s2).st_ino  # untouched shard hard-linked
        with safe_open(os.path.join(dst, "model-00001-of-00002.safetensors"), "pt") as f:
            for n, w in a.items():
                if n + "_scale" in f.keys():
                    q, s = f.get_tensor(n), f.get_tensor(n + "_scale")
                    dq = dq_mxfp8(q, s) if s.dtype == torch.uint8 else dq_fp8blk(q, s)
                    assert torch.equal(dq, w.float()), n  # on-grid inputs decode bit-exactly
                else:
                    assert bitwise_equal(f.get_tensor(n), w), n
            assert f.get_tensor(p1 + "self_attn.indexer.wq_b.weight").dtype == torch.bfloat16
            assert f.get_tensor(p1 + "self_attn.kv_b_proj.weight").dtype == torch.bfloat16
            assert f.get_tensor(p2 + "self_attn.q_a_proj.weight").dtype == torch.bfloat16  # MTP untouched
        cfg = json.load(open(os.path.join(dst, "config.json")))["quantization_config"]
        ql = cfg["quantized_layers"]
        assert cfg["quant_algo"] == "MIXED_PRECISION" and "config_groups" not in cfg
        assert ql[p0 + "self_attn.in_proj_qkvbfg_a"]["quant_algo"] == "MXFP8"
        assert ql[p1 + "self_attn.fused_qkv_a_proj"]["quant_algo"] == "FP8_PB_WO"
        assert ql[p1 + "mlp.shared_experts.gate_proj"]["quant_algo"] == "FP8_PB_WO"
        assert ql[p1 + "mlp.experts"] == {"quant_algo": "NVFP4", "group_size": 16}
        assert p1 + "self_attn.indexer.wq_b" not in ql and p1 + "self_attn.kv_b_proj" not in ql
        assert cfg["ignore"] == ["lm_head", p1 + "mlp.gate", "model.language_model.layers.2*"], cfg["ignore"]
        hf = json.load(open(os.path.join(dst, "hf_quant_config.json")))["quantization"]
        assert hf["quantized_layers"] == ql and hf["kv_cache_quant_algo"] == "FP8"
        wm = json.load(open(os.path.join(dst, "model.safetensors.index.json")))["weight_map"]
        assert wm[p0 + "self_attn.q_proj.weight_scale"] == "model-00001-of-00002.safetensors"
        assert os.path.exists(os.path.join(dst, "tokenizer.json"))
        # Dry run on the same source: headers only, same plan.
        assert len(convert(src, None, dry_run=True, tps=(2,))) == len(groups)
        # 6. Drafter: every decoder linear goes to FP8_PB_WO without a gate; fc stays BF16.
        dsrc, ddst = os.path.join(tmp, "dsrc"), os.path.join(tmp, "ddst")
        os.makedirs(dsrc)
        d = {f"layers.0.{m}.weight": (torch.randn(*sh, generator=g) * 0.02).to(torch.bfloat16) for m, sh in [
            ("self_attn.q_proj", (256, 256)), ("self_attn.k_proj", (256, 256)), ("self_attn.v_proj", (256, 256)),
            ("self_attn.o_proj", (256, 256)), ("mlp.gate_proj", (512, 256)), ("mlp.up_proj", (512, 256)),
            ("mlp.down_proj", (256, 512))]}
        d["fc.weight"] = torch.randn(256, 512, generator=g).to(torch.bfloat16)
        save_file(d, os.path.join(dsrc, "model.safetensors"))
        dump({"num_hidden_layers": 1}, os.path.join(dsrc, "config.json"))
        dg = convert(dsrc, ddst, drafter=True, tps=(2,))
        assert [x["status"] for x in dg] == ["converted"] * 4
        dq = json.load(open(os.path.join(ddst, "config.json")))["quantization_config"]
        assert dq == {"quant_method": "modelopt", "quant_algo": "FP8_PB_WO", "ignore": DRAFTER_IGNORE}, dq
        with safe_open(os.path.join(ddst, "model.safetensors"), "pt") as f:
            assert f.get_tensor("layers.0.mlp.down_proj.weight").dtype == torch.float8_e4m3fn
            assert f.get_tensor("fc.weight").dtype == torch.bfloat16
    print("self-test OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", nargs="?")
    ap.add_argument("dst", nargs="?")
    ap.add_argument("--dry-run", action="store_true", help="read headers only; print the plan and byte savings")
    ap.add_argument("--include", help="regex; only groups whose 'kind module members' string matches")
    ap.add_argument("--exclude", help="regex; drop matching groups (knapcio set: 'indexer\\.wq_b|f_b_proj|g_b_proj')")
    ap.add_argument("--tp", default="2,4", help="TP sizes every converted layer must shard evenly over")
    ap.add_argument("--max-rel-err", type=float, default=0.005, help="gate: max relative Frobenius error")
    ap.add_argument("--min-exact", type=float, default=0.5, help="gate: min fraction of bit-exact elements")
    ap.add_argument("--link", choices=("hard", "sym", "copy"), default="hard", help="how untouched files are placed")
    ap.add_argument("--drafter", action="store_true", help="DFlash2 drafter: q/k/v/o + gate/up/down to FP8_PB_WO")
    ap.add_argument("--self-test", action="store_true")
    a = ap.parse_args()
    if a.self_test:
        return self_test()
    if not a.src or not (a.dst or a.dry_run):
        ap.error("SRC and DST are required (DST optional with --dry-run)")
    convert(a.src, a.dst, drafter=a.drafter, dry_run=a.dry_run, include=a.include, exclude=a.exclude,
            tps=tuple(int(t) for t in a.tp.split(",")), max_rel=a.max_rel_err, min_exact=a.min_exact, link=a.link)


if __name__ == "__main__":
    main()
