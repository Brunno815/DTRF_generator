#!/bin/bash

# 1 = dataset
# 2 = G for golden, A for approximate
# 3 = classifier
# 4 = maxDepth
# 5 = numTrees
# 6 = bits_precision
# 7 = minPeriod
# 8 = period
# 9 = dirSynth

period=$8
minPeriod=$7
dirSynth=${9}

mkdir -p "SYNTH_REPORTS"

echo "SYNTH WITH PERIOD ${period}"

echo "Starting synthesis..."

cd ${dirSynth}/ && ./run_synth_vhdl -p ${period}

echo "Starting simulation..."
cd -

cd ${dirSynth}/ && ./run_sim_vhdl -A synthesis_RTL_ARCH_DECISION_MVT_false_VDD_1.0_HAFA_true_EFFORT_HHH_C${period} -s -t -v -l $1

echo "Starting resynthesis..."
cd -

cd ${dirSynth}/ && ./run_synth_vhdl -r -p ${period}

cd -

cp -r "${dirSynth}/synth/synthesis_RTL_ARCH_DECISION_MVT_false_VDD_1.0_HAFA_true_EFFORT_HHH_C${period}/reports" "./SYNTH_REPORTS/reports_$1_$2_$3_maxDepth_$4_numTrees_$5_bitsPrecision_$6_period_${period}"
python parse_power.py $1 $2 $3 $4 $5 ${minPeriod} ${period} $6
