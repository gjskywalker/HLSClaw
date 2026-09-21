#!/usr/bin/env python3
"""Dynamic (behavioral) Tier of HLS-LeVeri Dual-Tier Consistency Checking.

Reconstructs the "Behavioral Consistency (Runtime Equivalence)" stage: after the
static precondition holds, both programs are executed on identical stimuli and
their I/O traces are compared (paper Eq. 1: forall x, M_c(x) == M_h(x); Dynamic
Rate = N_match / N_total). The paper drives this with KLEE-derived stimuli and
cycle-accurate Vitis co-simulation. Here we substitute concrete differential
testing at the C level: the original C (P_c) and HLS-C (P_h) are each wrapped in
their own namespace, compiled together against the bundled HLS stubs, and run on
shared random inputs, comparing return values and array (pointer) side effects.

This is a functional-equivalence reconstruction, not RTL cycle-accurate co-sim.
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import tempfile

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUBS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "stubs"))
sys.path.insert(0, SCRIPT_DIR)
from static_consistency import dump_ast  # noqa: E402


# --------------------------------------------------------------------------- #
# Signature extraction from the clang AST
# --------------------------------------------------------------------------- #
def find_top(ast, name):
    """Return (return_type, [param_qualtypes]) for the top function `name`."""
    stack = [ast]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            if (node.get("kind") in ("FunctionDecl", "CXXMethodDecl")
                    and node.get("name") == name
                    and any(isinstance(c, dict) and c.get("kind") == "CompoundStmt"
                            for c in node.get("inner", []) or [])):
                qual = (node.get("type") or {}).get("qualType", "")
                ret = qual.split("(")[0].strip() or "void"
                params = [c["type"]["qualType"]
                          for c in node.get("inner", []) or []
                          if isinstance(c, dict) and c.get("kind") == "ParmVarDecl"]
                return ret, params
            stack.extend(node.get("inner", []) or [])
    raise RuntimeError(f"top function '{name}' with a body not found")


def classify(qualtype):
    """Return ('ptr'|'scalar', element_or_scalar_type, is_float)."""
    t = qualtype.strip()
    is_ptr = "*" in t or "[" in t
    base = re.sub(r"\[.*?\]", "", t).replace("*", "").strip()
    base = base.replace("const", "").replace("restrict", "").strip() or "int"
    is_float = "float" in base or "double" in base
    return ("ptr" if is_ptr else "scalar"), base, is_float


# --------------------------------------------------------------------------- #
# Source wrapping: hoist #include lines, wrap the rest in a namespace
# --------------------------------------------------------------------------- #
def wrap_namespace(src_path, ns):
    includes, body = [], []
    with open(src_path, "r", errors="replace") as fh:
        for line in fh:
            (includes if line.lstrip().startswith("#include") else body).append(line)
    return "".join(includes), f"namespace {ns} {{\n{''.join(body)}\n}}\n"


# --------------------------------------------------------------------------- #
# Driver generation
# --------------------------------------------------------------------------- #
def build_driver(ret, params, top, buf_size, trials, seed):
    decls, ref_args, dut_args, fill, cmp_ = [], [], [], [], []
    for i, p in enumerate(params):
        kind, base, is_float = classify(p)
        if kind == "ptr":
            decls.append(f"    {base} r{i}[N], d{i}[N];")
            gen = "(uniform() * 200.0 - 100.0)" if is_float else "(long)(uniform()*200-100)"
            fill.append(f"      {{ {base} _v = {gen}; r{i}[k] = _v; d{i}[k] = _v; }}")
            ref_args.append(f"r{i}")
            dut_args.append(f"d{i}")
            if is_float:
                cmp_.append(f"      if (std::fabs((double)r{i}[k]-(double)d{i}[k])>1e-6) ok=false;")
            else:
                cmp_.append(f"      if ((long long)r{i}[k]!=(long long)d{i}[k]) ok=false;")
        else:
            if is_float:
                decls.append(f"    {base} s{i} = ({base})(uniform()*200.0-100.0);")
            else:
                # clamp integer scalars into [0, N] (index/size-like inputs)
                decls.append(f"    {base} s{i} = ({base})(uniform()*(N+1));")
            ref_args.append(f"s{i}")
            dut_args.append(f"s{i}")

    ref_call = f"ref::{top}({', '.join(ref_args)})"
    dut_call = f"dut::{top}({', '.join(dut_args)})"
    if ret.replace(" ", "") in ("void", ""):
        call = f"    {ref_call};\n    {dut_call};"
        ret_cmp = ""
    else:
        call = f"    {ret} rr = {ref_call};\n    {ret} dd = {dut_call};"
        if "float" in ret or "double" in ret:
            ret_cmp = "    if (std::fabs((double)rr-(double)dd)>1e-6) ok=false;"
        else:
            ret_cmp = "    if ((long long)rr!=(long long)dd) ok=false;"

    return f"""
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstdint>
static unsigned long long _st;
static double uniform() {{ _st = _st*6364136223846793005ULL+1442695040888963407ULL;
    return ((_st>>11)&0x1FFFFFFFFFFFFFULL)/9007199254740992.0; }}
int main() {{
  const int N = {buf_size};
  _st = {seed}ULL;
  int match = 0;
  for (int t = 0; t < {trials}; ++t) {{
    bool ok = true;
{os.linesep.join(decls)}
    for (int k = 0; k < N; ++k) {{
{os.linesep.join(fill) if fill else "      ;"}
    }}
{call}
{ret_cmp}
    for (int k = 0; k < N; ++k) {{
{os.linesep.join(cmp_) if cmp_ else "      ;"}
    }}
    if (ok) ++match;
  }}
  std::printf("[leveri_dynamic_match]=%d/%d\\n", match, {trials});
  return 0;
}}
"""


def run(original, optimized, top, buf_size=64, trials=200, seed=1, timeout=120,
        include_dirs=None):
    ret, params = find_top(dump_ast(optimized, include_dirs), top)

    inc_c, ns_c = wrap_namespace(original, "ref")
    inc_h, ns_h = wrap_namespace(optimized, "dut")
    driver = build_driver(ret, params, top, buf_size, trials, seed)

    workdir = tempfile.mkdtemp(prefix="leveri_dyn_")
    tu = os.path.join(workdir, "leveri_diff.cpp")
    with open(tu, "w") as fh:
        fh.write(inc_c + inc_h + ns_c + ns_h + driver)

    binp = os.path.join(workdir, "leveri_diff")
    inc_flags = ["-I", STUBS_DIR]
    for d in (include_dirs or []):
        inc_flags += ["-I", d]
    comp = subprocess.run(["g++", "-w", "-O0", "-std=c++14", *inc_flags, tu,
                           "-o", binp], capture_output=True, text=True)
    if comp.returncode != 0:
        return {"status": "ERROR", "reason": "compile failed",
                "detail": comp.stderr[:1500], "tu": tu}
    try:
        exe = subprocess.run([binp], capture_output=True, text=True,
                             timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "reason": f"exceeded {timeout}s"}

    m = re.search(r"\[leveri_dynamic_match\]=(\d+)/(\d+)", exe.stdout)
    if not m:
        return {"status": "ERROR", "reason": "no result marker",
                "detail": (exe.stdout + exe.stderr)[:1500]}
    match, total = int(m.group(1)), int(m.group(2))
    rate = match / total if total else 0.0
    return {"status": "PASS" if match == total else "FAIL",
            "dynamic_rate": round(rate, 4), "match": match, "total": total}


def main():
    ap = argparse.ArgumentParser(
        description="HLS-LeVeri dynamic behavioral equivalence (Eq. 1).")
    ap.add_argument("optimized", help="HLS-oriented C/C++ (P_h) source")
    ap.add_argument("original", help="original plain C/C++ (P_c) source")
    ap.add_argument("function", help="top function name")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--buf-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        res = run(args.optimized, args.original, args.function, args.buf_size,
                  args.trials, args.seed, args.timeout, args.include)
    except Exception as exc:
        res = {"status": "ERROR", "reason": str(exc)}

    if args.json:
        print(json.dumps(res))
    else:
        for k in ("dynamic_rate", "match", "total"):
            if k in res:
                print(f"[leveri_{k}]={res[k]}")
        print(f"[leveri_dynamic_status]={res['status']}")
        if res.get("reason"):
            print(f"[leveri_dynamic_reason]={res['reason']}")
        if res.get("detail"):
            print(res["detail"])
    return 0 if res["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
