#!/usr/bin/env python3
"""Static Tier of HLS-LeVeri Dual-Tier Consistency Checking.

Reconstructs the static "Testbench Consistency (Static Precondition)" stage of
the paper *Shift-Left High-Level Synthesis Verification via Knowledge-Augmented
LLM Agent*. Given an original C source (P_c) and its HLS-oriented counterpart
(P_h / HLS-C), it computes three structural consistency scores over their ASTs
and aggregates them into S_static:

    S_IO   (Eq. 2)  1 - Dist_Lev(S_c, S_h) / max(|S_c|, |S_h|)
                    over linearized AST node sequences of data-injection
                    sub-trees (variable declarations, literals, array inits).
    S_CFG  (Eq. 3)  w_ctrl * Sim_ctrl + w_nest * Sim_nest
                    control-node distribution similarity + nesting-depth match.
    S_DDG  (Eq. 4)  |Ec ∩ Eh| / |Ec ∪ Eh|
                    Jaccard over def-use (reaching-definition) edges.
    S_static (Eq. 5) w1*S_IO + w2*S_CFG + w3*S_DDG  >=  tau_static (default 0.75)

The paper uses libclang/KLEE; here we drive the installed `clang` with
`-ast-dump=json` (no libclang-python dependency) and parse the JSON AST. HLS-only
headers are resolved through the bundled `stubs/` include directory so that
ap_int / hls::stream code parses. This is a reconstruction from the paper, not
the authors' original source.
"""
import argparse
import json
import os
import subprocess
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUBS_DIR = os.path.normpath(os.path.join(SCRIPT_DIR, "..", "stubs"))

CONTROL_KINDS = {
    "IfStmt", "ForStmt", "WhileStmt", "DoStmt", "SwitchStmt", "CXXForRangeStmt",
}
# AST sub-tree roots that carry input/data-injection structure (Eq. 2).
IO_ROOT_KINDS = {"VarDecl", "InitListExpr"}


# --------------------------------------------------------------------------- #
# Clang AST acquisition
# --------------------------------------------------------------------------- #
def dump_ast(path, extra_include_dirs=None, std="c++14"):
    """Return the clang JSON AST for `path`, or raise RuntimeError."""
    includes = ["-I", STUBS_DIR]
    for d in (extra_include_dirs or []):
        includes += ["-I", d]
    cmd = [
        "clang", "-Xclang", "-ast-dump=json", "-fsyntax-only",
        "-w", "-x", "c++", f"-std={std}", *includes, path,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    # clang still emits a (partial) AST on non-fatal diagnostics; only treat an
    # empty stdout as a hard failure.
    if not proc.stdout.strip():
        raise RuntimeError(
            f"clang produced no AST for {path}:\n{proc.stderr[:2000]}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensive
        raise RuntimeError(f"could not parse clang AST JSON for {path}: {exc}")


def iter_nodes(node):
    """Pre-order traversal over the clang JSON AST."""
    if not isinstance(node, dict):
        return
    yield node
    for child in node.get("inner", []) or []:
        yield from iter_nodes(child)


def collect_bodies(ast, src_path):
    """CompoundStmt bodies of functions *defined in the main source file*.

    clang emits declarations from every included header (including our HLS
    stubs, whose templates own method bodies). We track the current source file
    across the top-level declaration list and keep only functions whose
    declaration lives in `src_path`, so stub/system code never enters a score.
    """
    target = os.path.basename(src_path)
    bodies, current_file = [], None
    for decl in ast.get("inner", []) or []:
        if not isinstance(decl, dict):
            continue
        loc = decl.get("loc") or {}
        if loc.get("file"):
            current_file = loc["file"]
        if current_file is None or os.path.basename(current_file) != target:
            continue
        for node in iter_nodes(decl):
            if node.get("kind") in ("FunctionDecl", "CXXMethodDecl"):
                for child in node.get("inner", []) or []:
                    if isinstance(child, dict) and child.get("kind") == "CompoundStmt":
                        bodies.append(child)
    return bodies


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def levenshtein(a, b):
    """Standard edit distance between two sequences (two-row DP)."""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cost = 0 if ca == cb else 1
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost))
        prev = cur
    return prev[-1]


def ref_names(node):
    """Names of every DeclRefExpr within `node`'s subtree."""
    names = []
    for n in iter_nodes(node):
        if n.get("kind") == "DeclRefExpr":
            ref = n.get("referencedDecl") or {}
            name = ref.get("name")
            if name:
                names.append(name)
    return names


# --------------------------------------------------------------------------- #
# S_IO  (Eq. 2)
# --------------------------------------------------------------------------- #
def linearize_io(bodies):
    """Pre-order node-kind sequence of all data-injection sub-trees."""
    seq = []
    for body in bodies:
        for node in iter_nodes(body):
            if node.get("kind") in IO_ROOT_KINDS:
                seq.extend(n.get("kind", "?") for n in iter_nodes(node))
    return seq


def s_io(bodies_c, bodies_h):
    s_c = linearize_io(bodies_c)
    s_h = linearize_io(bodies_h)
    denom = max(len(s_c), len(s_h))
    if denom == 0:
        return 1.0  # nothing to inject on either side -> trivially aligned
    return 1.0 - levenshtein(s_c, s_h) / denom


# --------------------------------------------------------------------------- #
# S_CFG  (Eq. 3)
# --------------------------------------------------------------------------- #
def control_distribution(bodies):
    counts = {k: 0 for k in CONTROL_KINDS}
    for body in bodies:
        for node in iter_nodes(body):
            k = node.get("kind")
            if k in counts:
                counts[k] += 1
    return counts


def max_nesting(bodies):
    def depth(node, cur):
        best = cur
        add = 1 if node.get("kind") in CONTROL_KINDS else 0
        for child in node.get("inner", []) or []:
            if isinstance(child, dict):
                best = max(best, depth(child, cur + add))
        return best

    return max((depth(b, 0) for b in bodies), default=0)


def s_cfg(bodies_c, bodies_h, w_ctrl=0.5, w_nest=0.5):
    dc, dh = control_distribution(bodies_c), control_distribution(bodies_h)
    # Sim_ctrl: 1 - normalized L1 distance between control-node distributions.
    total = sum(dc.values()) + sum(dh.values())
    if total == 0:
        sim_ctrl = 1.0
    else:
        l1 = sum(abs(dc[k] - dh[k]) for k in CONTROL_KINDS)
        sim_ctrl = 1.0 - l1 / total

    # Sim_nest: normalized closeness of maximum control nesting depth.
    nc, nh = max_nesting(bodies_c), max_nesting(bodies_h)
    if max(nc, nh) == 0:
        sim_nest = 1.0
    else:
        sim_nest = 1.0 - abs(nc - nh) / max(nc, nh)

    return w_ctrl * sim_ctrl + w_nest * sim_nest, sim_ctrl, sim_nest


# --------------------------------------------------------------------------- #
# S_DDG  (Eq. 4)
# --------------------------------------------------------------------------- #
def ddg_edges(bodies):
    """Def-use edge set: (defined_var -> used_var) for assignments and inits."""
    edges = set()
    for body in bodies:
        for node in iter_nodes(body):
            kind = node.get("kind")
            if kind == "BinaryOperator" and node.get("opcode") == "=":
                inner = node.get("inner", []) or []
                if len(inner) >= 2:
                    lhs = ref_names(inner[0])
                    rhs = ref_names(inner[1])
                    if lhs:
                        for use in rhs:
                            edges.add((lhs[0], use))
            elif kind == "VarDecl" and (node.get("inner") or []):
                defname = node.get("name")
                if defname:
                    for use in ref_names(node):
                        if use != defname:
                            edges.add((defname, use))
    return edges


def s_ddg(bodies_c, bodies_h):
    ec = ddg_edges(bodies_c)
    eh = ddg_edges(bodies_h)
    union = ec | eh
    if not union:
        return 1.0
    return len(ec & eh) / len(union)


# --------------------------------------------------------------------------- #
# Aggregate + CLI
# --------------------------------------------------------------------------- #
def evaluate(original, optimized, weights=(1 / 3, 1 / 3, 1 / 3),
             include_dirs=None):
    bodies_c = collect_bodies(dump_ast(original, include_dirs), original)
    bodies_h = collect_bodies(dump_ast(optimized, include_dirs), optimized)

    v_io = s_io(bodies_c, bodies_h)
    v_cfg, sim_ctrl, sim_nest = s_cfg(bodies_c, bodies_h)
    v_ddg = s_ddg(bodies_c, bodies_h)

    w1, w2, w3 = weights
    v_static = w1 * v_io + w2 * v_cfg + w3 * v_ddg
    return {
        "s_io": round(v_io, 4),
        "s_cfg": round(v_cfg, 4),
        "sim_ctrl": round(sim_ctrl, 4),
        "sim_nest": round(sim_nest, 4),
        "s_ddg": round(v_ddg, 4),
        "s_static": round(v_static, 4),
    }


def main():
    ap = argparse.ArgumentParser(
        description="HLS-LeVeri static Dual-Tier consistency (Eq. 2-5).")
    ap.add_argument("optimized", help="HLS-oriented C/C++ (P_h) source")
    ap.add_argument("original", help="original plain C/C++ (P_c) source")
    ap.add_argument("--tau_static", type=float, default=0.75,
                    help="static consistency threshold (default 0.75)")
    ap.add_argument("--weights", default="0.34,0.33,0.33",
                    help="w1,w2,w3 for S_IO,S_CFG,S_DDG")
    ap.add_argument("--include", action="append", default=[],
                    help="extra include dir(s) for clang")
    ap.add_argument("--json", action="store_true", help="emit JSON only")
    args = ap.parse_args()

    weights = tuple(float(x) for x in args.weights.split(","))
    if len(weights) != 3:
        print("[leveri_static_status]=ERROR")
        print("[leveri_static_reason]=weights must be w1,w2,w3")
        return 2

    try:
        scores = evaluate(args.optimized, args.original, weights, args.include)
    except Exception as exc:  # surface clang/AST failures as ERROR, not crash
        print("[leveri_static_status]=ERROR")
        print(f"[leveri_static_reason]={exc}")
        return 2

    passed = scores["s_static"] >= args.tau_static
    scores["tau_static"] = args.tau_static
    scores["status"] = "PASS" if passed else "FAIL"

    if args.json:
        print(json.dumps(scores))
    else:
        for key in ("s_io", "s_cfg", "sim_ctrl", "sim_nest", "s_ddg",
                    "s_static", "tau_static"):
            print(f"[leveri_{key}]={scores[key]}")
        print(f"[leveri_static_status]={scores['status']}")
        if not passed:
            print("[leveri_static_reason]=S_static below tau_static "
                  "(structural precondition not met)")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
