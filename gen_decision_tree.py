import math
import sys
from os import system
from math import ceil
from itertools import combinations
import numpy as np
import pandas as pd

#def print_file(fileHandler, codeLine):
	#print(codeLine, file=fileHandler)

def gen_register(fileHandler, _input, _output, size):
	print("process(RST, CLK)", file=fileHandler)
	print("begin", file=fileHandler)
	print("\tif(RST = '1') then", file=fileHandler)
	print("\t\t%s <= %s;" % (_output, "'0'" if size == 1 else "(OTHERS => '0')"), file=fileHandler)
	print("\telsif(CLK'event and CLK = '1') then", file=fileHandler)
	print("\t\t%s <= %s;" % (_output, _input), file=fileHandler)
	print("\tend if;", file=fileHandler)
	print("end process;", file=fileHandler)
	print("\n\n", file=fileHandler)

def gen_comparator(fileHandler, input1, input2, output):
	print("process(%s, %s)" % (input1, input2), file=fileHandler)
	print("begin", file=fileHandler)
	print("\tif (%s < %s) then" % (input1, input2), file=fileHandler)
	print("\t\t%s <= '1';" % (output), file=fileHandler)
	print("\telsif (%s = %s) then" % (input1, input2), file=fileHandler)
	print("\t\t%s <= '1';" % (output), file=fileHandler)
	print("\telse", file=fileHandler)
	print("\t\t%s <= '0';" % (output), file=fileHandler)
	print("\tend if;", file=fileHandler)
	print("end process;", file=fileHandler)
	print("\n\n", file=fileHandler)


def gen_maj(fileHandler, inputs, outputs):

	for _inputs, _output in zip(inputs, outputs):

		nrInputsMaj = len(_inputs)
		sizeTermMaj = int(ceil(nrInputsMaj/2.0))

		expr = ""
		nands = []

		for comb in combinations(_inputs, sizeTermMaj):
			nands.append("not(%s)" % (" and ".join(comb)))

		expr = "%s <= not(%s);" % (_output, " and ".join(nands))

		print("\n", file=fileHandler)
		print("%s" % (expr), file=fileHandler)
		print("\n\n", file=fileHandler)


def gen_libraries(fileHandler, customLibraries = []):
	print("library IEEE;", file=fileHandler)
	print("use IEEE.STD_LOGIC_1164.ALL;", file=fileHandler)
	print("use IEEE.STD_LOGIC_UNSIGNED.ALL;", file=fileHandler)

	for customLibrary in customLibraries:
		print("use %s;" % (customLibrary), file=fileHandler)

	print("\n\n", file=fileHandler)


def gen_entity(fileHandler, nameEntity, inputs, outputs):
	print("entity %s is" % (nameEntity), file=fileHandler)
	print("\tPort(", file=fileHandler)
	
	for _input in inputs:
		if _input["size"] > 1:
			print("\t\t%s: in %s(%d downto 0);" % (_input["name"], _input["type"], _input["size"] - 1), file=fileHandler)
		else:
			print("\t\t%s: in %s;" % (_input["name"], _input["type"]), file=fileHandler)

	for idx, output in enumerate(outputs):
		if output["size"] > 1:
			print("\t\t%s: out %s(%d downto 0)%s" % (output["name"], output["type"], output["size"] - 1, ";" if idx != (len(outputs) - 1) else ""), file=fileHandler)
		else:
			print("\t\t%s: out %s%s" % (output["name"], output["type"], ";" if idx != (len(outputs) - 1) else ""), file=fileHandler)

	print("\t);", file=fileHandler)
	print("end %s;\n\n" % (nameEntity), file=fileHandler)


def gen_signals(fileHandler, signals):
	for signal in signals:
		#print signal['size']
		if signal["size"] > 1:
			print("signal %s: %s(%d downto 0)%s;" % (signal["name"], signal["type"], signal["size"]-1, " := \"%s\"" % (signal["const_value"]) if "const_value" in signal.keys() else ""), file=fileHandler)
		else:
			print("signal %s: %s%s;" % (signal["name"], signal["type"], " := \'%s\'" % (signal["const_value"]) if "const_value" in signal.keys() else ""), file=fileHandler)

	print("\n\n", file=fileHandler)


def gen_logic(fileHandler, attrs):
	for attr in attrs:
		print("%s <= %s %s %s;\n" % (attr[0], attr[1], attr[2], attr[3]), file=fileHandler)


def gen_ands_ors(fileHandler, exprs):
	for expr in exprs:
		print("%s <= %s;\n" % (expr["out"], (" %s " % (expr["op"])).join(expr["ops"])), file=fileHandler)


def gen_comps_priority(fileHandler, comps):
	for comp in comps:
		print("process(%s)" % (", ".join(comp['elems'])), file=fileHandler)
		print("begin", file=fileHandler)
		print("\tif (%s > %s) then" % (comp['elems'][0], comp['elems'][1]), file=fileHandler)
		sepNums = []
		for x in comp['elems']:
			sepNums += x.split("_")[1:]
		codeCat = "_".join(sepNums)
		print("\t\tcomp_add_%s <= '1';" % (codeCat), file=fileHandler)
		print("\t\tadd_%s <= %s;" % (codeCat, comp['elems'][0]), file=fileHandler)
		print("\telsif (%s = %s) then" % (comp['elems'][0], comp['elems'][1]), file=fileHandler)
		print("\t\tcomp_add_%s <= '1';" % (codeCat), file=fileHandler)
		print("\t\tadd_%s <= %s;" % (codeCat, comp['elems'][0]), file=fileHandler)
		print("\telse", file=fileHandler)
		print("\t\tcomp_add_%s <= '0';" % (codeCat), file=fileHandler)
		print("\t\tadd_%s <= %s;" % (codeCat, comp['elems'][1]), file=fileHandler)
		print("\tend if;", file=fileHandler)
		print("end process;", file=fileHandler)
		print("", file=fileHandler)


def gen_muxSels(fileHandler, muxSels):
	for muxSel in muxSels:
		print("with %s select" % (muxSel['selector']), file=fileHandler)
		if len(muxSel['decider']) > 1:
			print("\t%s <= '1' when \"%s\"," % (muxSel['output'], muxSel['decider']), file=fileHandler)
		else:
			print("\t%s <= '1' when '%s'," % (muxSel['output'], muxSel['decider']), file=fileHandler)
		print("\t\t\t '0' when OTHERS;", file=fileHandler)
		print("", file=fileHandler)


def gen_architecture(fileHandler, classifier, nameEntity, signals, registers, comparators, exprs, maj, nr_out = 2, comps = [], muxSels = []):

	print("architecture Behavioral of %s is\n" % (nameEntity), file=fileHandler)

	gen_signals(fileHandler, signals)

	print("begin\n", file=fileHandler)

	for register in registers:
		gen_register(fileHandler, register["input"], register["output"], register["size"])

	for comparator in comparators:
		gen_comparator(fileHandler, comparator["in1"], comparator["in2"], comparator["out"])

	gen_ands_ors(fileHandler, exprs)

	if maj != []:
		gen_maj(fileHandler, maj['inputs'], maj['outputs'])

	if comps != []:
		gen_comps_priority(fileHandler, comps)

	if muxSels != []:
		gen_muxSels(fileHandler, muxSels)

	if classifier == 'tree':
		if nr_out == 2:
			print("reg_decision <= or_0;", file=fileHandler)
		else:
			for i in range(nr_out):
				print("reg_decision(%d) <= or_%d;" % (i, i), file=fileHandler)

	print("end Behavioral;", file=fileHandler)



def gen_tb(fileHandler, fileDebug, dataset, classifier, nameEntity, inputs, outputs):

	print(dataset)

	print("`timescale 1ns / 1ps\n", file=fileDebug)
	print("module tst;\n", file=fileDebug)
	print("\tparameter", file=fileDebug)
	print("\t\tINPUT_PATH=\"./\",", file=fileDebug)
	print("\t\tPERIOD=60,", file=fileDebug)

	print("`timescale 1ns / 1ps\n", file=fileHandler)
	print("module tst;\n", file=fileHandler)
	print("\tparameter", file=fileHandler)
	print("\t\tINPUT_PATH=\"./\",", file=fileHandler)
	print("\t\tPERIOD=60,", file=fileHandler)

	df = pd.read_csv("datasets/%s_preprocessed_approx.csv" % (dataset))
	lines = df.shape[0]

	df2 = pd.read_csv("TEST_DATA/%s/default/testData.txt" % (dataset))
	lines2 = df2.shape[0]

	print(lines2)

	print("\t\tQTD=%d;\n" % (lines2+1), file=fileDebug)

	print("\tparameter", file=fileDebug)
	print("\t\tKBITS=64;\n", file=fileDebug)

	print("\treg [KBITS-1:0] K;", file=fileDebug)

	print("\t\tQTD=%d;\n" % (lines), file=fileHandler)

	print("\tparameter", file=fileHandler)
	print("\t\tKBITS=64;\n", file=fileHandler)

	print("\treg [KBITS-1:0] K;", file=fileHandler)

	###### INSERT THE INPUTS AND OUTPUTS HERE ######
	list_ins = []
	for _input in inputs:
		if _input['name'] != 'RST' and _input['name'] != 'CLK':
			print("\twire [%d-1:0] %s;" % (_input['size'], _input['name']), file=fileHandler)
			print("\twire [%d-1:0] %s;" % (_input['size'], _input['name']), file=fileDebug)
			list_ins.append(_input['name'])

	list_outs = []
	for _output in outputs:
		print("\twire [%s-1:0] %s;\n" % (_output['size'], _output['name']), file=fileHandler)
		print("\twire [%s-1:0] %s;\n" % (_output['size'], _output['name']), file=fileDebug)
		list_outs.append(_output['name'])
	################################################

	print("\tinteger f, i;", file=fileDebug)

	if list_ins != []:
		print("\tINPUTS_ALL #(INPUT_PATH,QTD,KBITS) ROM1 (K,%s);" % (",".join(list_ins)), file=fileHandler)
		print("\tINPUTS_ALL #(INPUT_PATH,QTD,KBITS) ROM1 (K,%s);" % (",".join(list_ins)), file=fileDebug)
	print("\treg CLK;", file=fileHandler)
	print("\treg CLK;", file=fileDebug)

	if list_ins != []:
		print("\tdecision DUT(%s,CLK,1'b0,%s);" % (",".join(list_ins), ",".join(list_outs)), file=fileHandler)
		print("\tdecision DUT(%s,CLK,1'b0,%s);" % (",".join(list_ins), ",".join(list_outs)), file=fileDebug)
	else:
		print("\tdecision DUT(CLK,1'b0,%s);" % (",".join(list_outs)), file=fileHandler)
		print("\tdecision DUT(CLK,1'b0,%s);" % (",".join(list_outs)), file=fileDebug)

	print("\tinitial CLK = 1'b0;", file=fileHandler)
	print("\talways #(PERIOD/2) CLK = ~CLK;\n", file=fileHandler)

	print("\tinitial", file=fileHandler)
	print("\tbegin", file=fileHandler)
	print("\t@(posedge CLK)", file=fileHandler)
	print("\tfor (K=0; K <= QTD-1; K=K+1)", file=fileHandler)
	print("\tbegin", file=fileHandler)
	print("\t\t@(posedge CLK);", file=fileHandler)
	print("\tend", file=fileHandler)
	print("\t$finish;", file=fileHandler)
	print("end", file=fileHandler)
	print("endmodule\n", file=fileHandler)

	print("\tinitial CLK = 1'b0;", file=fileDebug)
	print("\talways #(PERIOD/2) CLK = ~CLK;\n", file=fileDebug)

	print("\tinitial", file=fileDebug)
	print("\tbegin", file=fileDebug)
	print("\t\ti = 0;", file=fileDebug)
	print('\t\tf = $fopen("predict.txt","w");', file=fileDebug)
	#print('\t\t@(posedge CLK);', file=fileDebug)
	print("\t\tfor (K=0; K <= QTD; K=K+1)", file=fileDebug)
	print("\t\tbegin", file=fileDebug)
	print("\t\t\t@(negedge CLK);", file=fileDebug)
	print("\t\t\ti = i + 1;", file=fileDebug)
	print("\t\t\tif (i >= 3)", file=fileDebug)
	print("\t\t\tbegin", file=fileDebug)
	print('\t\t\t\t$display("%b", decision);', file=fileDebug)
	print('\t\t\t\t$fwrite(f, "%b\\n", decision);', file=fileDebug)
	print("\t\t\tend", file=fileDebug)
	print("\t\tend", file=fileDebug)
	print("\t\t$fclose(f);", file=fileDebug)
	print("\t\t$finish;", file=fileDebug)
	print("\tend", file=fileDebug)
	print("endmodule\n", file=fileDebug)

	if list_ins != []:
		print("module INPUTS_ALL(K,%s);" % (",".join(list_ins)), file=fileHandler)

		print("\tparameter", file=fileHandler)
		print("\t\tINPUT_PATH = \"./\",", file=fileHandler)
		print("\t\tLINHAS = %d," % (lines), file=fileHandler)
		print("\t\tNBITS = 64;\n", file=fileHandler)

		print("\tinput wire [NBITS-1:0] K;", file=fileHandler)

		print("module INPUTS_ALL(K,%s);" % (",".join(list_ins)), file=fileDebug)

		print("\tparameter", file=fileDebug)
		print("\t\tINPUT_PATH = \"./\",", file=fileDebug)
		print("\t\tLINHAS = %d," % (lines2+1), file=fileDebug)
		print("\t\tNBITS = 64;\n", file=fileDebug)

		print("\tinput wire [NBITS-1:0] K;", file=fileDebug)

		for _input in inputs:
			if _input['name'] != 'RST' and _input['name'] != 'CLK':
				print("\toutput reg [%d-1:0] %s;" % (_input['size'], _input['name']), file=fileHandler)
				print("\toutput reg [%d-1:0] %s;" % (_input['size'], _input['name']), file=fileDebug)

		for _input in inputs:
			if _input['name'] != 'RST' and _input['name'] != 'CLK':
				print("\treg [%d-1:0] in_%s [0:LINHAS-1];" % (_input['size'], _input['name']), file=fileHandler)
				print("\treg [%d-1:0] in_%s [0:LINHAS-1];" % (_input['size'], _input['name']), file=fileDebug)

		print("\n\tinitial", file=fileHandler)
		print("\tbegin", file=fileHandler)

		print("\n\tinitial", file=fileDebug)
		print("\tbegin", file=fileDebug)

		for _input in inputs:
			if _input['name'] != 'RST' and _input['name'] != 'CLK':
				print("\t\t$readmemb({INPUT_PATH,\"/\",\"%s.txt\"},in_%s);" % (_input['name'], _input['name']), file=fileHandler)
				print("\t\t$readmemb({INPUT_PATH,\"/\",\"%s.txt\"},in_%s);" % (_input['name'], _input['name']), file=fileDebug)

		print("\tend\n", file=fileHandler)

		print("\talways @( K )", file=fileHandler)
		print("\tbegin", file=fileHandler)

		print("\tend\n", file=fileDebug)

		print("\talways @( K )", file=fileDebug)
		print("\tbegin", file=fileDebug)

		for _input in inputs:
			if _input['name'] != 'RST' and _input['name'] != 'CLK':
				print("\t\t%s = in_%s[K];" % (_input['name'], _input['name']), file=fileHandler)
				print("\t\t%s = in_%s[K];" % (_input['name'], _input['name']), file=fileDebug)

		print("\tend", file=fileHandler)
		print("endmodule", file=fileHandler)

		print("\tend", file=fileDebug)
		print("endmodule", file=fileDebug)


'''
with open("decision_tree.vhd", "w") as fHandler:
        
	nameEntity = "decision_tree"

	gen_libraries(fHandler)
	gen_entity(fHandler, nameEntity, inputs, outputs)
	gen_architecture(fHandler, nameEntity, signals, codes)

'''
