---
name: profiling
description: Identify Vitis HLS performance bottlenecks and hotspots, and report constraint conflicts. Use when profiling a Vitis HLS design.
---

# Profiling (Vitis)

## When to use this skill
Use this skill to analyze Vitis HLS designs for memory/compute/control bottlenecks and hotspot loops/functions.

## Prerequisites
- The HLS source compiles and synthesizes.
- Target constraints are provided (frequency/resource/latency).

## Inputs
- HLS source path
- Build/synthesis configuration
- Target constraints

## Command 

### 1. Generate TCL Script
```bash
python scripts/gen_tcl.py <top_file> <top_func> <set_part> <create_clock_period> [output_file]
```
**Parameters:**
- `<top_file>`: Top-level HLS source file (e.g., `atax.cc`)
- `<top_func>`: Top function name (e.g., `dut`)
- `<set_part>`: FPGA part number (e.g., `xcu280-fsvh2892-2L-e`)
- `<create_clock_period>`: Clock period (e.g., `100MHz` or `10ns`)
- `[output_file]`: Optional output TCL file (default: `gen.tcl`)

**Example:**
```bash
python scripts/gen_tcl.py atax.cc dut xcu280-fsvh2892-2L-e 100MHz gen.tcl
```

### 2. Run Vitis HLS
```bash
python scripts/run_vitis.py <tcl_file>
```
**Parameters:**
- `<tcl_file>`: Path to TCL script file

**Output:**
- SUCCESS or FAILED flag
- On failure: generates `vitis_hls.log` for debugging

**Example:**
```bash
python scripts/run_vitis.py gen.tcl
```

### 3. Parse Synthesis Report
```bash
python scripts/parse_csynth.py <current_run_dir>/dut/solution/syn/report/csynth.xml
```
**Parameters:**
- `<csynth.rpt>`: Path to synthesis report file

**Output:**
- Performance & resource analysis
- Bottleneck identification
- Loop optimization recommendations

**Example:**
```bash
python scripts/parse_csynth.py csynth.xml
```

### Full Workflow
```bash
# Step 1: Generate TCL script
python scripts/gen_tcl.py atax.cc dut xcu280-fsvh2892-2L-e 100MHz gen.tcl

# Step 2: Run Vitis HLS synthesis
python scripts/run_vitis.py gen.tcl

# Step 3: Parse results and identify bottlenecks
python scripts/parse_csynth.py csynth.xml
```

## Outputs
- Hot function/loop list
- Bottleneck type and explanation
- II/latency info for key loops
- Constraint conflicts summary

Currently no reference files are included.

## Steps
1. Run Vitis profiling.
2. Parse the profiling report.
3. Summarize hotspots, bottleneck types, and conflicts.

## Notes
- Profiling generates synthesis-only TCL by default; it does not export RTL unless a downstream stage explicitly requests export.
- The agent caches parsed profiling reports by code hash, top function, target part, and clock period so repeated runs can reuse prior results.

## Failure handling
- Profiling fails: record the error and return an unavailable status.

## Logs & metrics
- Profiling runtime
- Report path
- Parse success rate
