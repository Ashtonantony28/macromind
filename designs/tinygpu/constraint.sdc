# MacroMind tinygpu_top. Period from TG_CLOCK_PERIOD (configs.mk), default 5 ns.
current_design tinygpu_top
set clk_period 5.0
if { [info exists ::env(TG_CLOCK_PERIOD)] } { set clk_period $::env(TG_CLOCK_PERIOD) }
create_clock -name core_clock -period $clk_period [get_ports clk]
set non_clk_inputs [lsearch -inline -all -not -exact [all_inputs] [get_ports clk]]
set_input_delay  [expr 0.2 * $clk_period] -clock core_clock $non_clk_inputs
set_output_delay [expr 0.2 * $clk_period] -clock core_clock [all_outputs]
