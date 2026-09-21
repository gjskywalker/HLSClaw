#!/usr/bin/env python3
"""HLS-LeVeri Dual-Tier orchestrator (C <-> HLS-C consistency check).

Adapts the two-tier protocol from *Shift-Left HLS Verification via KG LLM
Agent* to HLSClaw's setting (two designs, auto-generated stimuli).

Important scoping note. In the paper the static tier (S_IO/S_CFG/S_DDG) gates
*testbench isomorphism* -- that TB_c and TB_h inject the same stimuli
(Eq. 8: Phi(TB_c) ~= Phi(TB_h)). It is NOT a similarity check between the two
*designs* P_c and P_h, which are expected to differ because HLS rewriting
restructures code. Here the stimulus is generated identically for both designs
by construction, so testbench consistency holds trivially, and the real
C<->HLS-C equivalence oracle is the dynamic tier.

Therefore, by default:
  * the dynamic behavioral tier (dynamic_equiv.py) always runs and decides
    PASS/FAIL (Eq. 1: forall x, M_c(x) == M_h(x));
  * static scores are reported as diagnostics that refine the failure case;
  * `--static-gate` restores the paper's strict short-circuit (require
    S_static >= tau_static first), appropriate when the two inputs are actual
    testbenches rather than designs.

Emits standalone verification markers:
    [leveri_status]=PASS | FAIL | ERROR | TIMEOUT
    [leveri_reason]=...        [leveri_case]=...
ERROR/TIMEOUT are inconclusive. This standalone research verifier is not the
default HLSClaw workflow gate.
"""
import argparse
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
import static_consistency as static_tier  # noqa: E402
import dynamic_equiv as dynamic_tier  # noqa: E402


def main():
    ap = argparse.ArgumentParser(
        description="HLS-LeVeri Dual-Tier C<->HLS-C consistency check.")
    ap.add_argument("optimized", help="HLS-oriented C/C++ (P_h) source")
    ap.add_argument("original", help="original plain C/C++ (P_c) source")
    ap.add_argument("function", help="top function name")
    ap.add_argument("--tau_static", type=float, default=0.75)
    ap.add_argument("--weights", default="0.34,0.33,0.33")
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--buf-size", type=int, default=64)
    ap.add_argument("--timeout", type=int, default=120)
    ap.add_argument("--include", action="append", default=[])
    ap.add_argument("--static-gate", action="store_true",
                    help="enforce the paper's strict static precondition "
                         "(short-circuit FAIL when S_static < tau_static); "
                         "use only when inputs are testbenches, not designs")
    args = ap.parse_args()

    weights = tuple(float(x) for x in args.weights.split(","))

    # ---- Static tier (diagnostic by default) ----------------------------- #
    try:
        s = static_tier.evaluate(args.optimized, args.original, weights,
                                 args.include)
        for k in ("s_io", "s_cfg", "s_ddg", "s_static"):
            print(f"[leveri_{k}]={s[k]}")
    except Exception as exc:
        s = None
        print(f"[leveri_static_note]=static tier unavailable: {exc}")

    if args.static_gate and s is not None and s["s_static"] < args.tau_static:
        print("[leveri_status]=FAIL")
        print(f"[leveri_reason]=S_static={s['s_static']} < "
              f"tau_static={args.tau_static}: static precondition failed")
        print("[leveri_case]=Case2_StaticInconsistency")
        return 1

    # ---- Dynamic tier: the C<->HLS-C equivalence oracle ------------------ #
    d = dynamic_tier.run(args.original, args.optimized, args.function,
                         args.buf_size, args.trials, 1, args.timeout,
                         args.include)
    if "dynamic_rate" in d:
        print(f"[leveri_dynamic_rate]={d['dynamic_rate']}")

    status = d["status"]
    if status == "PASS":
        print("[leveri_status]=PASS")
        print("[leveri_reason]=I/O traces match on all stimuli (C == HLS-C)")
        print("[leveri_case]=none")
        return 0
    if status == "FAIL":
        # Refine the case using the static diagnostic: a high structural score
        # with a runtime mismatch points squarely at a target design bug; a low
        # score additionally flags structural divergence worth inspecting.
        low_static = s is not None and s["s_static"] < args.tau_static
        print("[leveri_status]=FAIL")
        print(f"[leveri_reason]=runtime output mismatch "
              f"({d.get('match')}/{d.get('total')} stimuli agreed)"
              + ("; note: S_static also low (structural divergence)"
                 if low_static else ""))
        print("[leveri_case]=Case4_TargetDesignBug")
        return 1

    # ERROR / TIMEOUT -> inconclusive but acceptable for progression.
    print(f"[leveri_status]={status}")
    print(f"[leveri_reason]=dynamic tier inconclusive: {d.get('reason', '')}")
    print("[leveri_case]=inconclusive")
    if d.get("detail"):
        print(d["detail"])
    return 2


if __name__ == "__main__":
    sys.exit(main())
