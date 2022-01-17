from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sys import argv
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import graphviz 
import os
from sklearn.tree import _tree
import copy
from copy import deepcopy
import gen_decision_tree as vhdl
import sys
import boolean
from ctes_trees import *
import pandas as pd
from itertools import product
import sklearn.datasets as ds
from math import log
from math import ceil
from math import sqrt
from math import floor
from preprocessing import *
from sklearn.tree import *
from collections import Counter

def genInputsDebug(dataset, orderedFeats, inputs, testData):
	folderDebugTest = 'TEST_DATA'
	system('mkdir -p %s/%s/%s' % (folderDebugTest, dataset, 'default'))

	df = pd.DataFrame()

	for feat in orderedFeats:
		df[feat] = testData[feat].head(maxLinesDebug)

	df.to_csv("%s/%s/%s/testData.txt" % (folderDebugTest, dataset, 'default'), index=False)

	for _input in inputs:
		if _input['name'] != 'CLK' and _input['name'] != 'RST':
			localDf = testData[_input['name']].head(maxLinesDebug)
			listBins = []
			strFormat = "{0:0%db}" % (int(_input['size']))
			for idx, elem in localDf.items():
				listBins.append(strFormat.format(int(elem)))
			df2 = pd.Series(listBins)
			df2.to_csv("%s/%s/%s.txt" % (folderDebugTest, dataset, _input['name']), index=False)


def debugClf(clf, classifier, dataset, orderedFeats, inputs, fileName, testData, testLabels, predictedByPy):

	np.savetxt("predictedByPy.txt", predictedByPy[:maxLinesDebug].astype(int), fmt='%d')

	os.system('rm -rf %s/TEST_DATA/*' % (dirSynth))
	os.system('cp -r TEST_DATA %s/' % (dirSynth))

	os.system('rm -rf %s/rtl/ARCH_DECISION/*' % (dirSynth))
	os.system('rm -rf %s/bench/*' % (dirSynth))

	os.system('cp %s/%s/%s.vhd %s/rtl/ARCH_DECISION/decision.vhd' % (folder_results, dirVhdl[classifier]['approx'], fileName, dirSynth))

	os.system('cp %s/%s/tb_debug_%s.v %s/bench/tb_decision.v' % (folder_results, dirVhdl[classifier]['approx'], fileName, dirSynth))

	os.system("bash exec_debug.sh 10000.0000 %s" % (dataset))

	os.system("cp %s/work/RTL_ARCH_DECISION_C_%s/predict.txt ." % (dirSynth, dataset))

	ypred_hdl = []
	with open("predict.txt","r") as fVhdl:
		lines = fVhdl.readlines()
		for line in lines:
			lLine = list(line.strip())
			if len(lLine) > 1:
				ypred_hdl.append(len(lLine) - 1 - lLine.index('1'))
			else:
				ypred_hdl.append(lLine[0])

	ypred_hdl = np.array(ypred_hdl).astype(int)

	isVhdlCorrect = np.array_equal(ypred_hdl, predictedByPy[:maxLinesDebug])

	return isVhdlCorrect


def gen_dataset_input_power(dataset, inputs):
	folder_inputs_power = "dataset_inputs_power"
	system("mkdir -p %s" % (folder_inputs_power))
	system("mkdir -p %s/%s" % (folder_inputs_power, dataset))

	df = pd.read_csv("datasets/%s_preprocessed_approx.csv" % (dataset))

	np.random.seed(0)
	df = df.sample(frac=1)

	if df.shape[0] > 100000:
		df = df.head(100000)

	for _input in inputs:
		if _input['name'] != 'CLK' and _input['name'] != 'RST':
			listBins = []
			strFormat = "{0:0%db}" % (int(_input['size']))

			for idx, elem in df[_input['name']].items():
				listBins.append(strFormat.format(int(elem)))

			df2 = pd.Series(listBins)
			df2.to_csv("%s/%s/%s.txt" % (folder_inputs_power, dataset, _input['name']), index = False, header = False)

	system("rm -rf %s/OUTPUT_DATASETS" % (dirSynth))
	system("mkdir -p %s/OUTPUT_DATASETS" % (dirSynth))
	system("cp -r %s/%s %s/OUTPUT_DATASETS/" % (folder_inputs_power, dataset, dirSynth))


def remove_feats(feats, list_remove, trainData, testData):
	for rem_feat in list_remove:
		if rem_feat in feats:
			feat_idx = feats.index(rem_feat)
			trainData = np.delete(trainData, [feat_idx], axis = 1)
			testData = np.delete(testData, [feat_idx], axis = 1)
			feats.remove(rem_feat)

	return feats, trainData, testData


def get_all_comparisons_ors(tree, featureSizes, which_out = -1):
	tree_ = tree.tree_
	featureName = [featureSizes[i][0] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
	featureSize = [featureSizes[i][1] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]

	comparisons = []

	ors = []

	def recurse(node, depth, expression):
		indent = "\t" * depth
		if tree_.feature[node] != _tree.TREE_UNDEFINED:
			name = featureName[node]
			threshold = tree_.threshold[node]
			_type = int(featureSize[node])

			if [name, threshold] not in comparisons:
				comparisons.append([name, threshold, _type])

			recurse(tree_.children_left[node], depth + 1, deepcopy(expression + [[name, threshold, _type, 'true']]))

			recurse(tree_.children_right[node], depth + 1, deepcopy(expression + [[name, threshold, _type, 'false']]))
		else:
			if which_out == -1:
				if np.argmax(tree_.value[node]) == 1:
					ors.append(deepcopy(expression))
			else:
				if np.argmax(tree_.value[node]) == which_out:
					ors.append(deepcopy(expression))

	recurse(0, 1, [])

	return comparisons, ors


def get_simplified_expression(tree, c, comparisons, ors):

	algebra = boolean.BooleanAlgebra()

	##### PARSE THE CONDITIONS INTO 'c%d' VALUES TO SIMPLIFY THEM USING BOOLEAN LIBRARY #####
	strExpr = ""

	if ors == []:
		isUsedInExpr = [0]*len(comparisons)
		return "0", isUsedInExpr
	
	for idxOr, elemOr in enumerate(ors):

		for idxAnd, elemAnd in enumerate(elemOr):

			idxElem = comparisons.index(elemAnd[:-1])
			notSymbol = "~" if elemAnd[3] == 'false' else ""
			condition = "%sc%s" % (notSymbol, idxElem)

			if idxOr == (len(ors) - 1):
				if len(elemOr) > 1:
					if idxAnd == 0 and idxAnd != (len(elemOr)-1):
						strExpr += '(%s&' % (condition)
					elif idxAnd == (len(elemOr)-1):
						strExpr += '%s)' % (condition)
					else:
						strExpr += '%s&' % (condition)
				else:
					strExpr += '%s' % (condition)
			else:
				if len(elemOr) > 1:
					if idxAnd == 0 and idxAnd != (len(elemOr)-1):
						strExpr += '(%s&' % (condition)
					elif idxAnd == (len(elemOr)-1):
						strExpr += '%s)|' % (condition)
					else:
						strExpr += '%s&' % (condition)
				else:
					strExpr += '%s|' % (condition)

	#########################################################################################
	simplExpr = algebra.parse(strExpr).simplify()

	##### ISUSEDINEXPR IS NEEDED TO CHECK LATER WHICH OF THE COMPARISONS ARE NOT USED IN THE EXPRESSION (THEREFORE, ONLY LEAD TO '0' LEAVES) #####
	isUsedInExpr = [1]*len(comparisons)

	for idx, comparison in enumerate(comparisons):
		if "c%s" % (idx) not in str(simplExpr):
			isUsedInExpr[idx] = 0
	##############################################################################################################################################

	return simplExpr, isUsedInExpr


def tree_to_hdl(clf, dataset, classifier, featureSizes, fHandler, fHandlerTb, fHandlerTbDebug, numTrees):

	nr_out = outs[dataset]

	##### CREATE OUTPUTS VECTOR WITH ONE OUTPUT (ONE BIT OF DECISION TREE, FIXED) #####
	outputs = []
	output = {}
	output['name'] = 'decision'
	output['size'] = 1 if (nr_out == 2 and classifier == 'tree') else nr_out
	output['type'] = 'STD_LOGIC' if (nr_out == 2 and classifier == 'tree') else 'STD_LOGIC_VECTOR'

	outputs.append(output)
	###################################################################################

	comparisons = {}
	ors = {}
	comparison = None
	_or = None
	c = {}

	for i in range(numTrees):
		comparisons[i] = []
		ors[i] = []
		for spec_out in range(nr_out):
			if classifier == 'tree':
				comparison, _or = get_all_comparisons_ors(clf, featureSizes, spec_out)
			elif classifier == 'forest':
				comparison, _or = get_all_comparisons_ors(clf[i], featureSizes, spec_out)

			ors[i].append(_or)
			
		comparisons[i] = comparison


	global_comparisons = []
	for i in range(numTrees):
		if comparisons[i] not in global_comparisons:
			global_comparisons += comparisons[i]

	algebra = boolean.BooleanAlgebra()

	c = [0]*numTrees
	expressions = {}
	isUsedInExprs = {}
	isUsedInExprs_per_tree = {}
	strExprs = {}

	for i in range(numTrees):

		expressions[i] = []
		isUsedInExprs[i] = []
		isUsedInExprs_per_tree[i] = [0]*len(global_comparisons)
		strExprs[i] = []

		for spec_out in range(nr_out):
			c[i] = [0]*len(global_comparisons)
			for idx in range(len(global_comparisons)):
				c[i][idx] = algebra.symbols("c%d" % (idx))


			expression = []
			isUsedInExpr = []
			if classifier == 'tree':
				expression, isUsedInExpr = get_simplified_expression(clf, c[i], global_comparisons, ors[i][spec_out])
			elif classifier == 'forest':
				expression, isUsedInExpr = get_simplified_expression(clf[i], c[i], global_comparisons, ors[i][spec_out])

			expressions[i].append(expression)
			isUsedInExprs[i].append(isUsedInExpr)
			isUsedInExprs_per_tree[i] = [(x or y) for x,y in zip(isUsedInExprs_per_tree[i], isUsedInExpr)]
			strExprs[i].append(str(expression))

	isUsedInExprs_global = [0]*len(isUsedInExprs_per_tree[0])

	for key in isUsedInExprs_per_tree.keys():
		for idx, elem in enumerate(isUsedInExprs_per_tree[key]):
			isUsedInExprs_global[idx] = elem or isUsedInExprs_global[idx]

	##### CREATE INPUTS VECTOR WHOSE INPUTS ARE THE FEATURES USED IN THE TREE #####

	inputs = []
	existingInputs = []

	for i in range(numTrees):
		for idx, comparison in enumerate(global_comparisons):
			if comparison[0] not in existingInputs and isUsedInExprs_global[idx] == 1:
				existingInputs.append(comparison[0])
				_input = {}
				_input['name'] = comparison[0]
				_input['type'] = 'STD_LOGIC' if comparison[2] == 1 else 'STD_LOGIC_VECTOR'
				_input['size'] = comparison[2]
				inputs.append(_input)

	_input = {}
	_input['name'] = "CLK"
	_input['type'] = "STD_LOGIC"
	_input['size'] = 1
	inputs.append(deepcopy(_input))

	_input['name'] = "RST"
	_input['type'] = "STD_LOGIC"
	_input['size'] = 1
	inputs.append(deepcopy(_input))
	###############################################################################

	##### WRITE THE LIBRARY NAMES AND THE ENTITY, WITH INPUTS AND OUTPUTS #####
	#print inputs
	vhdl.gen_libraries(fHandler)
	vhdl.gen_entity(fHandler, nameEntity, inputs, outputs)
	###########################################################################

	##### GET COMPARATORS AND SIGNALS #####

	signals = []
	comparators = []
	existingComparisons = []

	for _input in inputs:
		signal = {}
		signal["name"] = "reg_%s" % (_input["name"])
		signal["size"] = _input["size"]
		signal["type"] = _input["type"]
		signals.append(deepcopy(signal))

	signal = {}
	signal["name"] = "reg_%s" % (outputs[0]["name"])
	signal["size"] = outputs[0]["size"]
	signal["type"] = outputs[0]["type"]
	signals.append(deepcopy(signal))

	for i in range(numTrees):
		for idx, comparison in enumerate(global_comparisons):
			filteredComparison = "_".join([comparison[0], str(int(comparison[1])).replace(".","_")])
			if isUsedInExprs_global[idx] == 1 and filteredComparison not in existingComparisons:

				existingComparisons.append(filteredComparison)

				signal = {}
				signal["name"] = "comp_%s" % (filteredComparison)
				signal["size"] = 1
				signal["type"] = "STD_LOGIC"
				signals.append(deepcopy(signal))

				if "const_%s_%s" % (str(int(comparison[1])).replace(".","_"), str(comparison[2])) not in [existingSignal["name"].replace(".","_") for existingSignal in signals]:
					signal = {}
					signal["name"] = "const_%s_%s" % (str(int(comparison[1])).replace(".","_"), str(comparison[2]))
					for _input in inputs:
						if _input['name'] == comparison[0]:
							signal['type'] = _input["type"]
							signal["size"] = _input["size"]
							signal["const_value"] = "%s" % (('{0:0%sb}' % (signal["size"])).format(int(comparison[1])).replace("-",""))
							signals.append(deepcopy(signal))

				comparator = {}
				comparator["in1"] = "reg_" + comparison[0]
				comparator["in2"] = "const_%s_%s" % (str(int(comparison[1])).replace(".","_"), str(comparison[2]))
				comparator["out"] = "comp_%s" % (filteredComparison)
				comparators.append(deepcopy(comparator))

	###################################################

	exprs = []

	if classifier == 'forest':
		sizeBitsAdd = int(ceil(log(numTrees+1,2)))
		nrCatBits = sizeBitsAdd - 1
		for _out in range(nr_out):
			signal = {}
			signal['name'] = 'add_%d' % (_out)
			signal['size'] = sizeBitsAdd
			signal['type'] = 'STD_LOGIC_VECTOR' if sizeBitsAdd > 1 else 'STD_LOGIC'
			signals.append(deepcopy(signal))

		for _out in range(nr_out):
			expr = {}
			expr['out'] = 'add_%d' % (_out)
			expr['op'] = '+'
			if nrCatBits > 1:
				expr['ops'] = ['("%s" & or_%d)' % ('0'*nrCatBits, x) for x in range(_out, nr_out*numTrees, nr_out)]
			else:
				expr['ops'] = ["('0' & or_%d)" % (x) for x in range(_out, nr_out*numTrees, nr_out)]
			exprs.append(deepcopy(expr))
		
		comps = []
		cmprsnsGlobal = []
		highestsGlobal = []
		groupings = 2
		l = ["add_%d" % (x) for x in range(nr_out)]
		cmprsns = [l[i:i+groupings] for i in range(0, len(l), groupings)]
		while len(cmprsns) > 1:
			nextCmprsns = []
			for elem in cmprsns:
				concatNrs = "_".join(["_".join(x.split("_")[1:]) for x in elem])
				if len(elem) < 2:
					nextCmprsns.append("add_%s" % (concatNrs))
					continue
				comp = {}
				comp['elems'] = elem

				highestsGlobal.append('add_%s' % (concatNrs))
				cmprsnsGlobal.append('comp_add_%s' % (concatNrs))
				idxs = []
	
				nextCmprsns.append("add_%s" % (concatNrs))
				comps.append(deepcopy(comp))
	
			cmprsns = [nextCmprsns[i:i+groupings] for i in range(0,len(nextCmprsns),groupings)]
	
		comp = {}
		comp['elems'] = cmprsns[0]
		comps.append(deepcopy(comp))
		cmprsnsGlobal.append('comp_add_%s' % ("_".join(["_".join(x.split("_")[1:]) for x in cmprsns[0]])))

		for cmprsn in cmprsnsGlobal:
			signal = {}
			signal['name'] = cmprsn
			signal['size'] = 1
			signal['type'] = 'STD_LOGIC'
			signals.append(deepcopy(signal))
		for highest in highestsGlobal:
			signal = {}
			signal['name'] = highest
			signal['size'] = sizeBitsAdd
			signal['type'] = 'STD_LOGIC_VECTOR' if sizeBitsAdd > 1 else 'STD_LOGIC'
			signals.append(deepcopy(signal))

		signal = {}
		signal['name'] = "_".join(cmprsnsGlobal[-1].split("_")[1:])
		signal['size'] = sizeBitsAdd
		signal['type'] = 'STD_LOGIC_VECTOR'
		signals.append(deepcopy(signal))

	##### GET REGISTERS #####

	registers = []

	for _input in inputs:
		if _input["name"] == "CLK" or _input["name"] == "RST":
			continue
		register = {}
		register["input"] = _input["name"]
		register["output"] = "reg_%s" % (_input["name"])
		register["size"] = _input["size"]
		registers.append(deepcopy(register))

	register = {}
	register["input"] = "reg_%s" % (outputs[0]["name"])
	register["output"] = outputs[0]["name"]
	register["size"] = outputs[0]["size"]

	registers.append(deepcopy(register))

	muxSels = []

	if classifier == 'forest':
		for _out in range(nr_out):
			elemsContainOut = []
			for elem in cmprsnsGlobal:
				if str(_out) in elem.split("_")[2:]:
					elemsContainOut.append(elem)
			expr = {}
			expr['out'] = "sel_decision_%d" % (_out)
			expr['op'] = '&'
			expr['ops'] = elemsContainOut
	
			signal = {}
			signal['name'] = "sel_decision_%d" % (_out)
			signal['size'] = len(elemsContainOut)
			signal['type'] = 'STD_LOGIC_VECTOR' if len(elemsContainOut) > 1 else "STD_LOGIC"
			signals.append(deepcopy(signal))
	
			muxSel = {}
			muxSel['selector'] = "sel_decision_%d" % (_out)
			muxSel['output'] = 'reg_decision(%d)' % (_out)
			seqDecider = ""
			for code in elemsContainOut:
				for comp in comps:
					cats = "comp_add_" + "_".join(comp['elems'][0].split("_")[1:] + comp['elems'][1].split("_")[1:])
					add0 = comp['elems'][0].split("_")[1:]
					add1 = comp['elems'][1].split("_")[1:]
					if code == cats:
						if str(_out) in add0:
							seqDecider += "1"
						else:
							seqDecider += "0"
			muxSel['decider'] = seqDecider
	
			exprs.append(deepcopy(expr))
			muxSels.append(deepcopy(muxSel))
	#########################

	##### GET 'AND' AND 'OR' SIGNALS (EXPRESSIONS) #####

	lastIdx = 0
	for i in range(numTrees):
		for spec_out in range(nr_out):
			for idx, or_ in enumerate(strExprs[i][spec_out].split("|")):
				if or_ == "0":
					signal = {}
					expr = {}
					expr["out"] = "and_%d" % (idx + lastIdx)
					expr["op"] = "and"
					expr["ops"] = ["'0'", "'0'"]

					exprs.append(deepcopy(expr))

					signal["name"] = "and_%d" % (idx + lastIdx)
					signal["size"] = 1
					signal["type"] = "STD_LOGIC"
					signals.append(deepcopy(signal))

					continue


				signal = {}
				expr = {}
				expr["out"] = "and_%d" % (idx + lastIdx)
				expr["op"] = "and"
				expr["ops"] = []

				for idComp, comparison in enumerate(global_comparisons):
					filteredComparison = "_".join([comparison[0], str(int(comparison[1])).replace(".","_")])
					if "c%d" % (idComp) in or_.strip("(").strip(")").split("&"):
						expr["ops"].append(deepcopy("comp_%s" % (filteredComparison)))
					elif "~c%d" % (idComp) in or_.strip("(").strip(")").split("&"):
						expr["ops"].append(deepcopy("not(comp_%s)" % (filteredComparison)))

				if expr["ops"] == []:
					expr["ops"] = ["'1'", "'1'"]

				exprs.append(deepcopy(expr))

				signal["name"] = "and_%d" % (idx + lastIdx)
				signal["size"] = 1
				signal["type"] = "STD_LOGIC"
				signals.append(deepcopy(signal))

			signal = {}
			signal["name"] = "or_%d" % (spec_out + i * nr_out)
			signal["size"] = 1
			signal["type"] = "STD_LOGIC"
			signals.append(deepcopy(signal))

			expr["out"] = "or_%d" % (spec_out + i * nr_out)
			expr["op"] = "or"
			expr["ops"] = []

			expr["ops"] = ["and_%d" % (x + lastIdx) for x in range(len(strExprs[i][spec_out].split("|")))]

			exprs.append(deepcopy(expr))

			lastIdx = lastIdx + idx + 1

	##### GEN MAJORITARY GATE FOR ALL THE ORS #####

	maj = {}
	maj['inputs'] = [0]*nr_out
	maj['outputs'] = ['reg_decision(%d)' % (x) for x in range(nr_out)]
	for i in range(nr_out):
		maj['inputs'][i] = []
		for j in range(numTrees):
			maj['inputs'][i].append('or_%d' % (i+j*nr_out))

	###############################################

	if numTrees == 1:
		comps = []
		maj = []
		muxSels = []

	maj = []

	vhdl.gen_architecture(fHandler, classifier, nameEntity, signals, registers, comparators, exprs, maj, nr_out, comps, muxSels)

	####################################

	return inputs, outputs


def orderCodeFeats(tree, features):
	tree_ = tree.tree_
	featureName = [features[i] if i != _tree.TREE_UNDEFINED else "undefined!" for i in tree_.feature]
	orderedFeats = []
	for feat in featureName:
		if feat != "undefined!":
			orderedFeats.append(int(feat.split("_")[1]))
	orderedFeats = list(dict.fromkeys(orderedFeats))
	
	return ['feat_%d' % (x) for x in sorted(orderedFeats)]


def orderList(features):

	features = list(dict.fromkeys(features))

	orderedFeats = []
	for feat in features:
		orderedFeats.append(int(feat.split("_")[1]))

	return ['feat_%d' % (x) for x in sorted(orderedFeats)]


def genOrderedFeats(clf, classifier, features, numTrees = 5):
	intOrderedFeats = []
	if classifier == 'tree':
		intOrderedFeats = orderCodeFeats(clf, features)
	elif classifier == 'forest':
		intOrderedFeats = []
		allOrderedFeats = []
		for i in range(numTrees):
			localOrderedFeats = orderCodeFeats(clf[i], features)
			allOrderedFeats.append(localOrderedFeats)
			intOrderedFeats += localOrderedFeats

		intOrderedFeats = orderList(intOrderedFeats)

	return intOrderedFeats


def readDataset(dataset):

	data_preprocess = np.genfromtxt("%s/%s_preprocessed.csv" % (folder_datasets, dataset), skip_header = 1, delimiter = ',')
	data_approx = np.genfromtxt("%s/%s_preprocessed_approx.csv" % (folder_datasets, dataset), skip_header = 1, delimiter = ',')
	features = open("%s/%s_preprocessed.csv" % (folder_datasets, dataset), 'r').readline().strip('\n').strip('#').split(',')
	df_preprocess = pd.DataFrame(data = np.c_[data_preprocess], columns = features)
	df_approx = pd.DataFrame(data = np.c_[data_approx], columns = features)

	return data_preprocess, data_approx, features, df_preprocess, df_approx


def initializeSetup(classifier):
	os.system('mkdir -p %s/%s' % (folder_results, dirVhdl[classifier]['golden']))
	os.system('mkdir -p %s/%s' % (folder_results, dirVhdl[classifier]['approx']))


def trnsfrToSynthEnv(nameCfg):
	os.system('rm -rf %s/synth/*' % (dirSynth))
	os.system('rm -rf %s/work/*' % (dirSynth))
	os.system('cp %s/%s/%s.vhd %s/rtl/ARCH_DECISION/decision.vhd' % (folder_results, dirVhdl[classifier]['approx'], nameCfg, dirSynth))
	os.system('cp %s/%s/tb_%s.v %s/bench/tb_decision.v' % (folder_results, dirVhdl[classifier]['approx'], nameCfg, dirSynth))


def genAllOutputFiles(classifier, numTrees, maxDepth, dataset, bits_precision, GClf, AClf, featureSizes, testData):

	nameCfg = "%s_numTrees_%d_maxDepth_%d_bitsPrecision_%d" % (nameDecisionClass[dataset], numTrees, maxDepth, bits_precision)

	AfileOutVhdl = None
	
	AfileOutVhdl = open("%s/%s/%s.vhd" % (folder_results, dirVhdl[classifier]['approx'], nameCfg), 'w')
	AfileTbVhdl = open("%s/%s/tb_%s.v" % (folder_results, dirVhdl[classifier]['approx'], nameCfg), 'w')
	AfileTbVhdlDebug = open("%s/%s/tb_debug_%s.v" % (folder_results, dirVhdl[classifier]['approx'], nameCfg), 'w')

	features = [x[0] for x in featureSizes]
	orderedFeats = genOrderedFeats(AClf, classifier, features[:-1], numTrees)
	print("%s: Generating VHDL file for Golden and Approximate models..." % (dataset))
	inputs, outputs = tree_to_hdl(AClf, dataset, classifier, featureSizes, AfileOutVhdl, AfileTbVhdl, AfileTbVhdlDebug, numTrees)
	genInputsDebug(dataset, orderedFeats, inputs, testData)

	vhdl.gen_tb(AfileTbVhdl, AfileTbVhdlDebug, dataset, classifier, nameEntity, inputs, outputs)

	gen_dataset_input_power(dataset, inputs)

	AfileOutVhdl.close()
	AfileTbVhdl.close()
	AfileTbVhdlDebug.close()
	
	return orderedFeats, inputs



def trainTestClf(classifier, numTrees, maxDepth, dataset, bits_precision, featureSizes):

	print("Running dataset %s, with %s, for %d tree(s), maximum depth of %d and %d bits of precision" % (dataset, classifier, numTrees, maxDepth, bits_precision))

	initializeSetup(classifier)
	
	nameCfg = "%s_numTrees_%d_maxDepth_%d_bitsPrecision_%d" % (nameDecisionClass[dataset], numTrees, maxDepth, bits_precision)
	
	##### READ FEATURES AND OUTPUT LABELS #####
	print("%s: Reading features and output labels..." % (dataset))
	data_preprocess, data_approx, features, df_preprocess, df_approx = readDataset(dataset)
	####################################################

	##### GENERATE TRAIN AND TEST DATA FOR GOLDEN (G) AND APPROXIMATE MODELS #####
	print("%s: Generate train and test data for golden and approximate model..." % (dataset))
	Gtrain, Gtest, Gtrain_labels, Gtest_labels = train_test_split(df_preprocess.iloc[:,:-1], df_preprocess.iloc[:,-1], test_size = 0.33, random_state = 42)
	Atrain, Atest, Atrain_labels, Atest_labels = train_test_split(df_approx.iloc[:,:-1], df_approx.iloc[:,-1], test_size = 0.33, random_state = 42)
	#############################################################

	##### RUN THE DECISION TREE ALGORITHM (OR AN ALTERNATIVE ONE, LIKE THE RANDOM FOREST) #####
	print("%s: Running classifier for golden and approximate models..." % (dataset))
	GClf = None
	AClf = None

	if classifier == 'tree':
		GClf = DecisionTreeClassifier(max_depth = maxDepth).fit(Gtrain, Gtrain_labels)
		AClf = DecisionTreeClassifier(max_depth = maxDepth).fit(Atrain, Atrain_labels)
	elif classifier == 'forest':
		GClf = []
		AClf = []
		numFeats = len(featureSizes[:-1])
		numFeatsSub = int(sqrt(numFeats))
		for i in range(numTrees):
			colsIdx = np.random.choice(range(numFeats), numFeats - numFeatsSub)
			GtrainSub = np.array(Gtrain)
			AtrainSub = np.array(Atrain)
			GtrainSub[:,colsIdx] = 1
			AtrainSub[:,colsIdx] = 1
			Gtree = DecisionTreeClassifier(max_depth = maxDepth).fit(GtrainSub, Gtrain_labels)
			Atree = DecisionTreeClassifier(max_depth = maxDepth).fit(AtrainSub, Atrain_labels)
			GClf.append(Gtree)
			AClf.append(Atree)
	#print cross_val_score(clf, XTrain, yTrain, cv = 5)
	###########################################################################################

	##### GENERATE HIT SCORE FOR GOLDEN AND APPROXIMATE MODELS #####
	print("%s: Generating hit score for golden and approximate models..." % (dataset))
	yPredictGG = None
	yPredictGA = None
	yPredictAA = None
	if classifier == 'tree':
		yPredictGG = GClf.predict(Gtest)
		yPredictGA = GClf.predict(Atest)
		yPredictAA = AClf.predict(Atest)
	elif classifier == 'forest':
		GGvotes = []
		GAvotes = []
		AAvotes = []
		for i in range(numTrees):
			GGvotes.append(GClf[i].predict(Gtest))
			GAvotes.append(GClf[i].predict(Atest))	
			AAvotes.append(AClf[i].predict(Atest))
		GGvotes = np.array(GGvotes).T
		GAvotes = np.array(GAvotes).T
		AAvotes = np.array(AAvotes).T

		#yPredictGG = np.round(GGvotes.sum(axis=1)/float(numTrees)).astype('int')
		#yPredictGA = np.round(GAvotes.sum(axis=1)/float(numTrees)).astype('int')
		#yPredictAA = np.round(AAvotes.sum(axis=1)/float(numTrees)).astype('int')

		yPredictGG = []
		yPredictGA = []
		yPredictAA = []

		i = 0
		for vote in GGvotes:
			srtdVote = sorted(vote)
			most_common = Counter(srtdVote).most_common()
			max_occ = max([x[1] for x in most_common])
			most_common = min([x[0] for x in most_common if x[1] == max_occ])
			yPredictGG.append(most_common)
		for vote in GAvotes:
			srtdVote = sorted(vote)
			most_common = Counter(srtdVote).most_common()
			max_occ = max([x[1] for x in most_common])
			most_common = min([x[0] for x in most_common if x[1] == max_occ])
			yPredictGA.append(most_common)
		for vote in AAvotes:
			srtdVote = sorted(vote)
			most_common = Counter(srtdVote).most_common()
			max_occ = max([x[1] for x in most_common])
			most_common = min([x[0] for x in most_common if x[1] == max_occ])
			yPredictAA.append(most_common)

		yPredictGG = np.array(yPredictGG)
		yPredictGA = np.array(yPredictGA)
		yPredictAA = np.array(yPredictAA)

	GGscore = accuracy_score(Gtest_labels, yPredictGG)
	GAscore = accuracy_score(Atest_labels, yPredictGA)
	AAscore = accuracy_score(Atest_labels, yPredictAA)

	print("accuracy golden train with golden test: %f" % (GGscore))
	print("accuracy golden train with approximate test: %f" % (GAscore))
	print("accuracy approximate train with approximate test: %f" % (AAscore))
	#############################################################################################

	return [GClf, AClf, GGscore, GAscore, AAscore, Atest, Atest_labels, yPredictAA]



np.random.seed(seed)

fHits = open("%s/acc_values.csv" % (folder_results), "a")
for classifier in classifiers:
	if classifier == "tree":
		numTrees = [1]

	for numTree, maxDepth, dataset, bits_precision in product(numTrees, maxDepths, datasets, bits_precisions):

		featureSizes = preprocessing("%s/%s.csv" % (folder_datasets, dataset), Nbits = bits_precision, drop_cols = drop_cols[dataset])

		np.random.seed(seed)
		GClf, AClf, GGHitValue, GAHitValue, AAHitValue, Atest, Atest_labels, yPredictAA = trainTestClf(classifier, numTree, maxDepth, dataset, bits_precision, featureSizes)

		orderedFeats, inputs = genAllOutputFiles(classifier, numTree, maxDepth, dataset, bits_precision, GClf, AClf, featureSizes, Atest)

		nameCfg = "%s_numTrees_%d_maxDepth_%d_bitsPrecision_%d" % (nameDecisionClass[dataset], numTree, maxDepth, bits_precision)

		isVhdlCorrect = 0
		isVhdlCorrect = debugClf(AClf, classifier, dataset, orderedFeats, inputs, nameCfg, Atest, Atest_labels, yPredictAA)

		print(",".join([str(x) for x in [dataset, bits_precision, classifier, maxDepth, numTree, GGHitValue, GAHitValue, AAHitValue, isVhdlCorrect]]), file=fHits)

		print("Correctness state of VHDL: %d" % (isVhdlCorrect))

		trnsfrToSynthEnv(nameCfg)

		for period in list_periods:
			os.system("bash exec_decision_spec_period.sh %s %s %s %s %s %s %s %s %s" % (dataset, 'A', classifier, str(maxDepth), str(numTree), bits_precision, "%.4f" % (basePeriod), period, dirSynth))

fHits.close()
