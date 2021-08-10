#### Template Script for RTL->Gate-Level Flow (generated from RC GENUS15.22 - 15.20-s024_1) 

if {[file exists /proc/cpuinfo]} {
  sh grep "model name" /proc/cpuinfo
  sh grep "cpu MHz"    /proc/cpuinfo
}

puts "Hostname : [info hostname]"

####################################################################
## Load Design
####################################################################

#### Including setup file (root attributes & setup variables).
include $::env(TCL_ROOT)/setup_template_lp.tcl

#
if {[expr {[info exists ::env(RESYNTHETIZE)] && [string is false -strict $::env(RESYNTHETIZE)]}]} {
    #read_hdl -vhdl -library STD_DT "$::env(CURRENT_HDL_FOLDER)/STD_DT.vhd"
    set hdl_file_list "$::env(CURRENT_HDL_FOLDER)/decision.vhd"
    read_hdl -vhdl $hdl_file_list
    #read_hdl -v2001 $::env(CURRENT_HDL_FOLDER)/decision.v
    #read_hdl -language vhdl -library STD_DT "$::env(CURRENT_HDL_FOLDER)/STD_DT.vhd"

    elaborate $DESIGN
    puts "Runtime & Memory after 'read_hdl'"
    time_info Elaboration

    check_design -unresolved -unloaded -undriven > $_REPORTS_PATH/elab/${DESIGN}_check_design.rpt
} else {
    read_netlist $::env(CURRENT_HDL_FILE)
}


####################################################################
## Constraints Setup
####################################################################

read_sdc "$::env(CONSTR_ROOT)/decision.sdc"

#### Read in CPF file.
#apply_power_intent
#report_low_power_intent

# Read RTL activity file
#if {[file exists "$::env(TCF_ROOT)/RTL_$::env(BASE_SIM_FILENAME)_$::env(RTL_VIDEO_SIM).tcf"]} {
#    puts "TCF file present! Using early power estimation from RTL simulation..."
#    puts "Reading TCF file: "
#    read_tcf "$::env(TCF_ROOT)/RTL_$::env(BASE_SIM_FILENAME)_$::env(RTL_VIDEO_SIM).tcf"
#} else {
#    puts "Skipping early power estimation..."
#}

###################################################################################
## Define cost groups (clock-clock, clock-output, input-clock, input-output)
###################################################################################
#path_adjust -delay -300 -to [find / -clock *] -name overcon

## Uncomment to remove already existing costgroups before creating new ones.
## rm [find /designs/* -cost_group *]

if {[llength [all::all_seqs]] > 0} {
    define_cost_group -name C2C -design $DESIGN
    path_group -from [all::all_seqs] -to [all::all_seqs] -group C2C -name C2C
}

#define_cost_group -name I2O -design $DESIGN
#path_group -from [all::all_inps]  -to [all::all_outs] -group I2O -name I2O

#write_snapshot -outdir $_REPORTS_PATH -tag initial
#report_summary -outdir $_REPORTS_PATH
#
foreach cg [find / -cost_group *] {
    report timing -cost_group [list $cg] >> $_REPORTS_PATH/elab/${DESIGN}_pretim_${cg}.rpt
}
report_timing -lint > $_REPORTS_PATH/elab/${DESIGN}_timing_lint.rpt
report_timing -worst 3 > $_REPORTS_PATH/elab/${DESIGN}_timing_nworst_3.rpt

#### Including LOW POWER setup. (Leakage/Dynamic power/Clock Gating setup)
include $::env(TCL_ROOT)/power_template_lp.tcl

#### uniquify the subdesign if it is multiple instantiated
#### and you would like to assign one of the instantiations
#### to a different library domain
#### edit_netlist uniquify <design|subdesign>
#check_cpf



#### To turn off sequential merging on the design 
#### uncomment & use the following attributes.
##set_attribute optimize_merge_flops false /
##set_attribute optimize_merge_latches false /
#### For a particular instance use attribute 'optimize_merge_seqs' to turn off sequential merging. 

#commit_power_intent
#verify_power_structure -lp_only -post_synth -detail > $_REPORTS_PATH/${DESIGN}_verify_power_struct.rpt


####################################################################################################
## Synthesizing to generic 
####################################################################################################
if {[expr {[info exists ::env(RESYNTHETIZE)] && [string is false -strict $::env(RESYNTHETIZE)]}]} {
    set_attribute syn_generic_effort $SYN_EFF /
    syn_generic
    puts "Runtime & Memory after 'syn_generic'"
    time_info GENERIC

    #write_snapshot -outdir $_REPORTS_PATH -tag generic
    report datapath > $_REPORTS_PATH/generic/${DESIGN}_datapath.rpt
    report_timing -lint > $_REPORTS_PATH/generic/${DESIGN}_timing_lint.rpt
    report_timing -worst 3 > $_REPORTS_PATH/generic/${DESIGN}_timing_nworst_3.rpt
    #report_summary -outdir $_REPORTS_PATH


    #### Build RTL power models
    ##build_rtl_power_models -design $DESIGN -clean_up_netlist [-clock_gating_logic] [-relative <hierarchical instance>]
    #build_rtl_power_models -design $DESIGN -clean_up_netlist 
    #report power -rtl


    ####################################################################################################
    ## Synthesizing to gates
    ####################################################################################################

    set_attribute syn_map_effort $MAP_EFF /
    syn_map
    puts "Runtime & Memory after 'syn_map'"
    time_info MAPPED
    #write_snapshot -outdir $_REPORTS_PATH -tag map
    #report_summary -outdir $_REPORTS_PATH
    report datapath > $_REPORTS_PATH/map/${DESIGN}_datapath.rpt
    report_timing -lint > $_REPORTS_PATH/map/${DESIGN}_timing_lint.rpt
    report_timing -worst 3 > $_REPORTS_PATH/map/${DESIGN}_timing_nworst_3.rpt

    #report_power -clock_tree > $_REPORTS_PATH/map/${DESIGN}_clocktree_power.rpt

    #foreach cg [find / -cost_group *] {
    #  foreach mode [find / -mode *] {
    #    report timing -cost_group [list $cg] -mode $mode >> $_REPORTS_PATH/${DESIGN}_[vbasename $cg]_post_map.rpt
    #  }
    #}


    ##Intermediate netlist for LEC verification..
    #write_hdl -lec > ${_OUTPUTS_PATH}/${DESIGN}_intermediate.v
    #set cpf_file [write_cpf -output_dir ${_OUTPUTS_PATH} -prefix ${DESIGN}_]
    #write_do_lec -revised_design ${_OUTPUTS_PATH}/${DESIGN}_intermediate.v -cpf_revised $cpf_file  -logfile ${_LOG_PATH}/rtl2intermediate.lec.log > ${_OUTPUTS_PATH}/rtl2intermediate.lec.do

    ## ungroup -threshold <value>

    #######################################################################################################
    ## Opt Synthesis
    #######################################################################################################

    ## Uncomment to remove assigns & insert tiehilo cells during Incremental synthesis
    ##set_attribute remove_assigns true /
    ##set_remove_assign_options -buffer_or_inverter <libcell> -design <design|subdesign> 
    ##set_attribute use_tiehilo_for_const <none|duplicate|unique> /
    set_attribute syn_opt_effort $MAP_EFF /
    syn_opt
    #write_snapshot -outdir $_REPORTS_PATH -tag syn_opt
    #report_summary -outdir $_REPORTS_PATH

    puts "Runtime & Memory after incremental synthesis"
    time_info INCREMENTAL

    #foreach cg [find / -cost_group *] {
    #  foreach mode [find / -mode *] {
    #    report timing -cost_group [list $cg] -mode $mode >> $_REPORTS_PATH/${DESIGN}_[vbasename $cg]_post_incr.rpt
    #  }
    #}

    #######################################################################################################
    ## Incremental Synthesis
    #######################################################################################################
     
    ## Uncomment to remove assigns & insert tiehilo cells during Incremental synthesis
    ##set_attribute remove_assigns true /
    ##set_remove_assign_options -buffer_or_inverter <libcell> -design <design|subdesign>
    ##set_attribute use_tiehilo_for_const <none|duplicate|unique> /
    puts "Fazer incremental"
    syn_opt -incremental
    puts "Fazer snapshot"
    write_snapshot -outdir $_REPORTS_PATH -tag syn_opt_incrementar
    #report_summary -outdir $_REPORTS_PATH
    puts "Runtime & Memory after incremental optimization synthesis"
    time_info INCREMENTAL_POST_SCAN_CHAINS



    #####################################################################################################
    ## QoS Prediction & Optimization.
    #####################################################################################################
    if {[expr {[info exists ::env(USE_INNOVUS)] && [string is true -strict $::env(USE_INNOVUS)]}]} {
        puts "Using Innovus physical flow"
        set_attribute invs_temp_dir ${_OUTPUTS_PATH}/genus_invs_pred /
        set_attribute syn_opt_effort $PHYS_EFF /
        syn_opt -physical
        puts "Runtime & Memory after physical synthesis"
        time_info PHYSICAL
    } else {
        puts "Skipping Innovus physical flow"
    }
}

# Read Activity file after synthesis
puts "Checking for input stimuli file"
if {[catch {set vcd_file_list [glob "$::env(VCD_ROOT)/SYNTH_$::env(BASE_SIM_FILENAME)*.vcd"]} err]} {
    puts "VCD files not present!"
    puts "Testing for TCF files..."
    
    if {[catch {set tcf_file_list [glob "$::env(TCF_ROOT)/SYNTH_$::env(BASE_SIM_FILENAME)*.tcf"]} err]} { 
        puts "TCF files not present!"
    } else {
        puts "Found TCF files!"
    }
}

if {[info exists vcd_file_list]} {
    puts "Starting accurate power estimation with VCD..."
    foreach current_vcd $vcd_file_list {
        set file_basename [file rootname [file tail $current_vcd]]
        set video [lindex [split $file_basename _] end]
        puts "Reading VCD file: ${file_basename}."
        read_vcd -static -vcd_scope $::env(TESTBENCH_INST) $current_vcd
        report_gates -power > ${_REPORTS_PATH}/${DESIGN}_${video}_gates_power.rpt
        report_power -verbose > ${_REPORTS_PATH}/${DESIGN}_${video}_power.rpt
        report_power -verbose -tcf_summary > ${_REPORTS_PATH}/${DESIGN}_${video}_tcf_summary.rpt
        report_qor -power -levels_of_logic > ${_REPORTS_PATH}/${DESIGN}_${video}_qor.rpt
        report clock_gating > $_REPORTS_PATH/${DESIGN}_${video}_clockgating.rpt
        report_power -clock_tree > $_REPORTS_PATH/${DESIGN}_${video}_clocktree_power.rpt
    }
} elseif {[info exists tcf_file_list]} { 
    puts "Starting accurate power estimation with TCF..."
    foreach current_tcf $tcf_file_list {
        set file_basename [file rootname [file tail $current_tcf]]
        #set video [lindex [split $file_basename _] end]
	set video [join [lrange [split $file_basename _] 12 end] "_"]
        puts "Reading TCF file: ${file_basename}."
        read_tcf $current_tcf
        report_gates -power > ${_REPORTS_PATH}/${DESIGN}_${video}_gates_power.rpt
        report_power -verbose > ${_REPORTS_PATH}/${DESIGN}_${video}_power.rpt
        report_power -verbose -tcf_summary > ${_REPORTS_PATH}/${DESIGN}_${video}_tcf_summary.rpt
        report_qor -power -levels_of_logic > ${_REPORTS_PATH}/${DESIGN}_${video}_qor.rpt
        report clock_gating > $_REPORTS_PATH/${DESIGN}_${video}_clockgating.rpt
        report_power -clock_tree > $_REPORTS_PATH/${DESIGN}_${video}_clocktree_power.rpt
    }
} else {
    puts "Starting tool-based power estimation"
    report_gates -power > ${_REPORTS_PATH}/${DESIGN}_gates_power.rpt
    report_power -verbose > ${_REPORTS_PATH}/${DESIGN}_power.rpt
    report_power -verbose -tcf_summary > ${_REPORTS_PATH}/${DESIGN}_tcf_summary.rpt
    report_qor -power -levels_of_logic > ${_REPORTS_PATH}/${DESIGN}_qor.rpt
    report clock_gating > $_REPORTS_PATH/${DESIGN}_clockgating.rpt
    #report_power -clock_tree > $_REPORTS_PATH/${DESIGN}_clocktree_power.rpt
}



######################################################################################################
## write backend file set (verilog, SDC, config, etc.)
######################################################################################################

# Timing Report
report_timing -worst 3 > $_REPORTS_PATH/${DESIGN}_timing_nworst_3.rpt
report_timing -lint -verbose > $_REPORTS_PATH/${DESIGN}_timing_lint.rpt
foreach cg [find / -cost_group *] {
  report timing -cost_group [list $cg] >> $_REPORTS_PATH/${DESIGN}_[vbasename $cg]_final.rpt
}

# Report design 
report area -physical -normalize_with_gate HS65_LS_NAND2X2 > $_REPORTS_PATH/${DESIGN}_area_NAND2_eq.rpt
report datapath > $_REPORTS_PATH/${DESIGN}_datapath_incr.rpt
report messages > $_REPORTS_PATH/${DESIGN}_messages.rpt
write_snapshot -outdir $_REPORTS_PATH -tag final
#report_summary -outdir $_REPORTS_PATH

#write_hdl -mapped  > ${_OUTPUTS_PATH}/${DESIGN}_m.v
## write_script > ${_OUTPUTS_PATH}/${DESIGN}_m.script
write_design  -innovus -base_name ${_OUTPUTS_PATH}/${DESIGN}
#write_netlist -mapped > ${_OUTPUTS_PATH}/${DESIGN}_2
#write_do_lec -no_exit -revised_design ${output_path}/${DESIGN}.v -logfile ${log_path}/final.out > $lec_path/${DESIGN}-final.do
#write_sdf -recrem merge_always -setuphold merge_always -edges check_edge > ${_OUTPUTS_PATH}/${DESIGN}.sdf
write_sdf > ${_OUTPUTS_PATH}/${DESIGN}.sdf
#foreach mode [find / -mode *] {
#  write_sdc -mode $mode > ${_OUTPUTS_PATH}/${DESIGN}_m_${mode}.sdc
#}


#################################
### write_do_lec
#################################


#write_do_lec -golden_design ${_OUTPUTS_PATH}/${DESIGN}_intermediate.v -revised_design ${_OUTPUTS_PATH}/${DESIGN}_m.v -logfile  ${_LOG_PATH}/intermediate2final.lec.log > ${_OUTPUTS_PATH}/intermediate2final.lec.do
##Uncomment if the RTL is to be compared with the final netlist..
##write_do_lec -revised_design ${_OUTPUTS_PATH}/${DESIGN}_m.v -logfile ${_LOG_PATH}/rtl2final.lec.log > ${_OUTPUTS_PATH}/rtl2final.lec.do

puts "Final Runtime & Memory."
time_info FINAL
puts "============================"
puts "Synthesis Finished ........."
puts "============================"

#file copy [get_attribute stdout_log /] ${_LOG_PATH}/.

quit
