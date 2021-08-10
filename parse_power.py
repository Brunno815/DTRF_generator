from sys import argv

name_dataset = argv[1]
g_a = argv[2]
type_decision = argv[3]
depth_decision = argv[4]
nr_trees_decision = argv[5]
min_period = argv[6]
period = argv[7]
bits_precision = argv[8]

name_arch = "decision"
folder_reports = "SYNTH_REPORTS"
spec_report = "_".join(["reports", name_dataset, g_a, type_decision, "maxDepth", depth_decision, "numTrees", nr_trees_decision, "bitsPrecision", bits_precision, "period", period])
name_power_default = "decision_power.rpt"
name_power_dataset = "decision_%s_power.rpt" % (name_dataset)
name_area = "final_area.rpt"
name_nand2 = "decision_area_NAND2_eq.rpt"
folder_results = "GEN_RESULTS"
leak = 0.0
dyn = 0.0
total = 0.0
cells = 0
cell_area = 0
total_area = 0
nand2_eq_total = 0

with open("%s/%s/%s" % (folder_reports, spec_report, name_power_default), "r") as f_power, open("%s/%s/%s" % (folder_reports, spec_report, name_power_dataset), "r") as f_power_dataset, open("%s/%s/%s" % (folder_reports, spec_report, name_area), "r") as f_area, open("%s/%s/%s" % (folder_reports, spec_report, name_nand2), "r") as f_nand2, open("%s/synthesis_results.csv" % (folder_results), "a") as f_out:
	for line in f_power.readlines():
		splitLine = line.split()
		if len(splitLine) > 0:
			if splitLine[0] == name_arch:
				leak = float(splitLine[2])
				dyn = float(splitLine[5])
				total = float(splitLine[6])

	for line in f_power_dataset.readlines():
		splitLine = line.split()
		if len(splitLine) > 0:
			if splitLine[0] == name_arch:
				leak_dataset = float(splitLine[2])
				dyn_dataset = float(splitLine[5])
				total_dataset = float(splitLine[6])

	for line in f_area.readlines():
		splitLine = line.split()
		if len(splitLine) > 0:
			if splitLine[0] == name_arch:
				cells = int(splitLine[1])
				cell_area = int(splitLine[2])
				total_area = int(splitLine[4])

	for line in f_nand2.readlines():
		splitLine = line.split()
		if len(splitLine) > 0:
			if splitLine[0] == name_arch:
				nand2_eq_total = int(splitLine[4])

	print("\t".join([name_dataset, g_a, type_decision, bits_precision, depth_decision, nr_trees_decision, str(min_period), str(period), str(leak), str(dyn), str(total), str(leak_dataset), str(dyn_dataset), str(total_dataset), str(cells), str(cell_area), str(total_area), str(nand2_eq_total)]), file=f_out)
