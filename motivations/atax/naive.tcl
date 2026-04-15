# Vitis HLS 2022.2
set top_file_name "[file normalize [file join [file dirname [info script]] naive.cc]]"
set top_func_name "dut"
open_project -reset atax_naive
set_top $top_func_name

add_files $top_file_name
open_solution solution -flow_target vivado

set_part xcu250-figd2104-2L-e

create_clock -period 5.000 -name default

config_compile -pipeline_loops 0
config_dataflow -strict_mode warning
set_clock_uncertainty 27.000000%
config_rtl -enable_maxiConservative=1

csynth_design

close_project
puts "HLS completed successfully"
exit
