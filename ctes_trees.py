from os import system
import os

datasets = ['accelAdl', 'activities', 'wearable', 'wireless']
datasets = ['wearable']

outs = {}
outs['accelAdl'] = 14
outs['wearable'] = 5
outs['activities'] = 5
outs['wireless'] = 4

folder_datasets = 'datasets'
folder_results = 'GEN_RESULTS'
dirSizes = 'bits_features'
curDir = os.getcwd()
dirSynth = '%s/SYNTH' % (curDir)

drop_cols = {}
drop_cols['accelAdl'] = None
drop_cols['activities'] = None
drop_cols["wearable"] = ['user_name', 'raw_timestamp_part_1', 'raw_timestamp_part_2', 'cvtd_timestamp']
drop_cols["gyroscope"] = ['Index','Arrival_Time','Creation_Time']
drop_cols["wireless"] = ['Time']

nameDecisionClass = {}
for dataset in datasets:
	nameDecisionClass[dataset] = dataset

nameFunction = 'decision'
nameEntity = 'decision'

dirVhdl = {}
dirVhdl['tree'] = {}
dirVhdl['forest'] = {}
dirVhdl['tree']['golden'] = 'VHDL_trees_golden'
dirVhdl['forest']['golden'] = 'VHDL_forests_golden'
dirVhdl['tree']['approx'] = 'VHDL_trees_approx'
dirVhdl['forest']['approx'] = 'VHDL_forests_approx'

system('mkdir -p %s' % (folder_results))

fPower = open("%s/synthesis_results.csv" % (folder_results), "w")
print("\t".join(["Dataset", "G/A", "Classifier", "Bits_Precision", "Depth", "NrTrees", "Min_period", "Period", "Leakage_PWR", "Dyn_PWR", "Total_PWR", "Leakage_PWR_dataset", "Dyn_PWR_dataset", "Total_PWR_dataset", "Cells", "Cell_Area", "Total_Area", "NAND2_Eq_Total"]), file=fPower)
fPower.close()

fHits = open("%s/acc_values.csv" % (folder_results), "w")
print(",".join(['Dataset', 'Precision', 'Classifier', 'MaxDepth', 'Simul', 'NumTrees', 'GGAcc', 'GAAcc', 'AAAcc', 'isVhdlCorrect']), file=fHits)
fHits.close()

seed = 0
maxLinesDebug = 100

bits_precisions = [4,8,16,32]
maxDepths = [1,2,4]
numTrees = [3,6,9,12]
classifiers = ['forest']
basePeriod = 10000.0

list_periods = ["%.4f" % (float(basePeriod)/x) for x in [1.0]]

classifiers = ['forest']

bits_precisions = [10]
maxDepths = [4]
numTrees = [3]
