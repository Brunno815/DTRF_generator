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

