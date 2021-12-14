# DT/RF VLSI Generator

## General Description
This project consists of a framework that integrates the Scikit-Learn structures of Decision Trees (DTs) and Random Forests (RFs), along with a dedicated HDL generation of RTL descriptions.

This project was developed so that designers can easily perform design space explorations regarding the possible models for their data sets. This framework was developed to work with data sets that contain well-defined features in CSV files (see examples in the data sets folder to generalize).

## Preprocessing

Preprocessing of the data is performed by normalizing the values of every feature according to their limit values (minimum and maximum). Then, quantization is performed based on the number of bits requested for the features. Features with less than 20 different values are considered as categorical; in these cases, only the required number of bits to represent the different values is considered. The data sets are then split into training and test sets using the train_test_split method of Scikit-Learn.

## Training

For the DTs, we use the native DecisionTreeClassifier structure from Scikit-Learn. For the RFs, we employ a handcrafted implementation so that we can have control of the majority voter and make it more hardware-friendly.

Then, the structure is translated to RTL descriptions using our dedicated translator.

## Requirements

This code uses Genus and Irun for the synthesis, both of which are from Cadence. Therefore, the respective licenses are required.

## Running the code

To run the code, you should run the train_data.py file, which will train the data sets, generate the RTL and synthesize it using Cadence tools. The ctes_trees.py file defines the configurations used in the DTs/RFs to be generated, as well as target period, data set, and so on.
