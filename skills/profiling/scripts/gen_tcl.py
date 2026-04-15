#!/usr/bin/env python3
"""
Generate Vivado HLS TCL scripts from Python parameters.
Generates complete TCL scripts for Vitis HLS 2022.2+
"""

from pathlib import Path
import re


def _source_uses_dataflow(top_file_name: str) -> bool:
    """Return True when the source already contains an explicit HLS DATAFLOW pragma."""
    try:
        code = Path(top_file_name).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return re.search(r"#\s*pragma\s+HLS\s+dataflow\b", code, re.IGNORECASE) is not None

def _append_config_command(tcl_lines, command_name, params, use_equals=False):
    if not params:
        return
    cmd = command_name
    for key, value in params.items():
        if use_equals:
            cmd += f" {key}={value}"
        else:
            cmd += f" {key} {value}"
    tcl_lines.append(cmd)


def generate_tcl_script(top_file_name, top_func_name, set_part, create_clock_period,
                       project_name=None, enable_csim=False, enable_cosim=False,
                       enable_export=False, clock_uncertainty="27.000000%", config_compile_params=None,
                       config_interface_params=None, config_dataflow_params=None, config_rtl_params=None,
                       config_schedule_params=None, config_op_params=None, directive_commands=None):
    """
    Generate a complete Vitis HLS TCL script.
    
    Args:
        top_file_name (str): Source file name (e.g., "atax.cc")
        top_func_name (str): Top function name (e.g., "dut")
        set_part (str): FPGA part number (e.g., "xcu280-fsvh2892-2L-e")
        create_clock_period (str): Clock period (e.g., "100MHz" or "10ns")
        project_name (str, optional): Project name. Defaults to top_func_name
        enable_csim (bool): Enable C simulation. Defaults to False
        enable_cosim (bool): Enable C/RTL co-simulation. Defaults to False
        enable_export (bool): Export RTL after synthesis. Defaults to False
        clock_uncertainty (str): Clock uncertainty percentage. Defaults to "27.000000%"
        config_compile_params (dict, optional): Additional config_compile parameters
        config_interface_params (dict, optional): Additional config_interface parameters
        config_dataflow_params (dict, optional): Additional config_dataflow parameters
        config_rtl_params (dict, optional): Additional config_rtl parameters
        config_schedule_params (dict, optional): Additional config_schedule parameters
        config_op_params (list, optional): List of config_op commands to add
        directive_commands (list, optional): Additional Tcl directives such as set_directive_* commands
    
    Returns:
        str: Complete TCL script
    """
    
    if project_name is None:
        project_name = top_func_name
    is_dataflow_design = _source_uses_dataflow(top_file_name)
    
    # Start building the TCL script
    tcl_lines = []
    
    # Header
    tcl_lines.append("# Vitis HLS 2022.2")
    tcl_lines.append(f'set top_file_name "{top_file_name}"')
    tcl_lines.append(f'set top_func_name "{top_func_name}"')
    tcl_lines.append(f"open_project {project_name}")
    tcl_lines.append(f"set_top $top_func_name")
    tcl_lines.append("")
    
    # Add files
    tcl_lines.append("add_files $top_file_name")
    tcl_lines.append("open_solution solution -flow_target vivado")
    tcl_lines.append("")
    
    # Set part
    tcl_lines.append(f"set_part {set_part}")
    tcl_lines.append("")
    
    # Create clock
    tcl_lines.append(f"create_clock -period {create_clock_period} -name default")
    tcl_lines.append("")
    
    # Config compile
    if config_compile_params is None:
        config_compile_params = {"-pipeline_loops": "0"}
    
    config_compile_cmd = "config_compile"
    for key, value in config_compile_params.items():
        config_compile_cmd += f" {key} {value}"
    if is_dataflow_design and str(config_compile_params.get("-pipeline_loops", "")).strip() == "0":
        tcl_lines.append("# Dataflow design detected: leave automatic loop pipelining enabled.")
        tcl_lines.append(f"# {config_compile_cmd}")
    else:
        tcl_lines.append(config_compile_cmd)

    # Config interface
    _append_config_command(tcl_lines, "config_interface", config_interface_params)
    
    # Config dataflow
    if config_dataflow_params is None:
        config_dataflow_params = {"-strict_mode": "warning"}
    
    _append_config_command(tcl_lines, "config_dataflow", config_dataflow_params)
    
    # Clock uncertainty
    tcl_lines.append(f"set_clock_uncertainty {clock_uncertainty}")
    
    # Config RTL
    if config_rtl_params is None:
        config_rtl_params = {"-enable_maxiConservative": "1"}
    
    _append_config_command(tcl_lines, "config_rtl", config_rtl_params, use_equals=True)

    # Config schedule
    _append_config_command(tcl_lines, "config_schedule", config_schedule_params)
    
    tcl_lines.append("")
    
    # Config operations (optional)
    if config_op_params:
        tcl_lines.append("# Operation bindings")
        for op_cmd in config_op_params:
            tcl_lines.append(op_cmd)
        tcl_lines.append("")

    if directive_commands:
        tcl_lines.append("# Additional directives")
        for directive_cmd in directive_commands:
            tcl_lines.append(directive_cmd)
        tcl_lines.append("")
    
    # Simulation and synthesis commands
    if enable_csim:
        tcl_lines.append("csim_design")
    else:
        tcl_lines.append("# csim_design")
    
    tcl_lines.append("csynth_design")
    
    if enable_cosim:
        tcl_lines.append("cosim_design -rtl verilog -setup")
    else:
        tcl_lines.append("# cosim_design -rtl verilog -setup")
    
    tcl_lines.append("")
    
    # Export design only when a downstream stage explicitly needs RTL artifacts.
    if enable_export:
        tcl_lines.append("export_design -rtl verilog -flow syn")
        tcl_lines.append("")
    
    # Close project
    tcl_lines.append("close_project")
    tcl_lines.append('puts "HLS completed successfully"')
    tcl_lines.append("exit")
    
    return "\n".join(tcl_lines)


def save_tcl_script(filename, tcl_script):
    """Save TCL script to file."""
    with open(filename, 'w') as f:
        f.write(tcl_script)
    print(f"TCL script saved to {filename}")


if __name__ == "__main__":
    import sys

    usage = "Usage: python gen_tcl.py <top_file> <top_func> <part> <clock_period> [output_file]"
    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        print(usage)
        raise SystemExit(0)
    if len(sys.argv) == 1:
        print(usage)
        raise SystemExit(1)
    if len(sys.argv) < 5:
        print(usage, file=sys.stderr)
        raise SystemExit(2)

    top_file = sys.argv[1]
    top_func = sys.argv[2]
    part = sys.argv[3]
    clock_period = sys.argv[4]
    output_file = sys.argv[5] if len(sys.argv) > 5 else "gen.tcl"

    tcl_script = generate_tcl_script(
        top_file_name=top_file,
        top_func_name=top_func,
        set_part=part,
        create_clock_period=clock_period
    )
    save_tcl_script(output_file, tcl_script)
