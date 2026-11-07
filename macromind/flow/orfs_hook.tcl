# MacroMind macro-placement hook for OpenROAD-flow-scripts.
#
# Used as ORFS's MACRO_PLACEMENT_TCL: ORFS's scripts/macro_place_util.tcl
# sources this file inside stage 2_2_floorplan_macro *instead of* calling
# rtl_macro_placer (it still runs placement_blockages.tcl / block_channels
# afterwards and writes 2_2_floorplan_macro.odb + .tcl itself).
#
# Steps: write the floorplan DEF -> run `python -m macromind.flow.place` in the
# MacroMind venv (torch lives there, not in OpenROAD's embedded Python 3.10) ->
# source the place_macro commands it wrote (macros become LOCKED/FIRM).
#
# Environment (set by the generated harness config.mk):
#   MACROMIND_METHOD   random|grid|sa|rl
#   MACROMIND_SEED     integer seed
#   MACROMIND_PYTHON   venv python
#   MACROMIND_ROOT     repo root (cwd + PYTHONPATH for the placer)
#   MACROMIND_OUT_DIR  where the DEF / TCL / JSON / log go
#   MACROMIND_BUDGET   optional placer budget
#   MACROMIND_RL_CKPT  optional RL checkpoint (absolute)

foreach v {MACROMIND_METHOD MACROMIND_SEED MACROMIND_PYTHON MACROMIND_ROOT MACROMIND_OUT_DIR} {
  if { ![info exists ::env($v)] || $::env($v) == "" } {
    error "MacroMind hook: environment variable $v is not set"
  }
}

set mm_dir $::env(MACROMIND_OUT_DIR)
file mkdir $mm_dir
set mm_def [file join $mm_dir 2_1_floorplan.def]
set mm_tcl [file join $mm_dir macro_placement.tcl]
set mm_json [file join $mm_dir placer.json]
set mm_log [file join $mm_dir placer.log]
file delete -force $mm_tcl $mm_json

write_def $mm_def

set mm_lefs [list $::env(TECH_LEF) $::env(SC_LEF)]
if { [info exists ::env(ADDITIONAL_LEFS)] && $::env(ADDITIONAL_LEFS) != "" } {
  foreach l $::env(ADDITIONAL_LEFS) { lappend mm_lefs $l }
}
set mm_halo 0
if { [info exists ::env(MACRO_PLACE_HALO)] && $::env(MACRO_PLACE_HALO) != "" } {
  set mm_halo [tcl::mathfunc::max {*}$::env(MACRO_PLACE_HALO)]
}

set mm_cmd [list env -u PYTHONHOME PYTHONPATH=$::env(MACROMIND_ROOT) \
  $::env(MACROMIND_PYTHON) -m macromind.flow.place \
  --def $mm_def --lef {*}$mm_lefs \
  --method $::env(MACROMIND_METHOD) --seed $::env(MACROMIND_SEED) --halo $mm_halo \
  --out $mm_tcl --json $mm_json]
if { [info exists ::env(MACROMIND_BUDGET)] && $::env(MACROMIND_BUDGET) != "" } {
  lappend mm_cmd --budget $::env(MACROMIND_BUDGET)
}
if { [info exists ::env(MACROMIND_RL_CKPT)] && $::env(MACROMIND_RL_CKPT) != "" } {
  lappend mm_cmd --checkpoint $::env(MACROMIND_RL_CKPT)
}

puts "MacroMind: [join $mm_cmd { }]"
set mm_t0 [clock milliseconds]
set mm_pwd [pwd]
cd $::env(MACROMIND_ROOT)
if { [catch { exec {*}$mm_cmd >& $mm_log } mm_err] } {
  cd $mm_pwd
  if { [file exists $mm_log] } {
    set fh [open $mm_log]; puts [read $fh]; close $fh
  }
  error "MacroMind placer failed: $mm_err"
}
cd $mm_pwd
set fh [open $mm_log]; puts [read $fh]; close $fh
puts "MacroMind: placer took [expr {([clock milliseconds] - $mm_t0) / 1000.0}] s"

if { ![file exists $mm_tcl] } {
  error "MacroMind placer wrote no placement ($mm_tcl)"
}
source $mm_tcl
puts "MacroMind: applied $mm_tcl"
