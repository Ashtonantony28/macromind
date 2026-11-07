# MacroMind label dump for ORFS stage 5_1_grt (training data for the predictors).
#
# Used as ORFS's PRE_GLOBAL_ROUTE hook (scripts/global_route.tcl sources it right
# after loading 4_cts.odb). It does not route anything itself: it wraps ORFS's
# `report_metrics` proc so that, when the final "global route" metrics are
# reported (after incremental repair, with `estimate_parasitics -global_routing`
# parasitics in place, i.e. exactly the state the ORFS GRT metrics describe),
# it first writes $REPORTS_DIR/macromind_labels/endpoint_slack.rpt
# (`report_checks -format end`: every constrained setup endpoint's slack).
#
# The per-gcell routing capacity/usage needs no dump: GRT stores it in the
# dbGCellGrid of 5_1_grt.odb (read by macromind/predict/odb_gcells.py).
# (gui::dump_heatmap would be simpler but fails headless with GUI-0072.)
#
# No ORFS file is patched.

if { [info commands mm_orig_report_metrics] == "" } {
  rename report_metrics mm_orig_report_metrics
  proc report_metrics { stage when args } {
    if { $when == "global route" } {
      if { [catch { mm_dump_labels } mm_err] } {
        puts "MacroMind labels: dump failed: $mm_err"
      }
    }
    uplevel 1 [list mm_orig_report_metrics $stage $when {*}$args]
  }
}

proc mm_dump_labels { } {
  set d [file join $::env(REPORTS_DIR) macromind_labels]
  file mkdir $d
  set t0 [clock milliseconds]
  report_checks -path_delay max -group_path_count 1000000 -endpoint_path_count 1 \
    -format end -digits 4 > [file join $d endpoint_slack.rpt]
  puts "MacroMind labels: wrote $d in [expr {([clock milliseconds] - $t0) / 1000.0}] s"
}
