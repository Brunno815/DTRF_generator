This code generates RTL descriptions from Scikit-Learn structures of Decision Trees (DTs) and Random Forests (RFs). For the DTs, we use native DecisionTreeClassifier structure from Scikit-Learn. For the RFs, we employ a handcrafted implementation so that we can have control of the majority voter and make it more hardware-friendly.

This code uses Genus and Irun for the synthesis, both of which are from Cadence. Therefore, licenses are required.

To run the code, you should run the train_data.py file, which will train the data sets, generate the RTL and synthesize it using Cadence tools. The ctes_trees.py file defines the configurations used in the DTs/RFs to be generated, as well as target period, data set, and so on.
