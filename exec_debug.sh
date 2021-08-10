#!/bin/bash

period=$1

dirSynth="`pwd`"

echo ${dirSynth}
echo "Starting debugging simulation..."

cd ${dirSynth}/SYNTH && ./run_sim_vhdl -a -v -l $2
