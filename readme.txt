COMPSCI 687: Course Project
Sankaran Vaidyanathan

# Requirements

Python 3.7 or higher, numpy, scipy, cma, tqdm, pandas

# Instructions

All commands are run from the source directory as root

- To generate the dataset, place data.csv in the source directory and run proc_data.py
This will output to dataset.pkl (the file will be over 2GB)

- To generate policies, run hcope.py
Policies will be stored in {root-dir}/policies/

- To test the generated policies, run eval_safety.py
Tested policies will be stored in {root-dir}/policies/checked
For the final submission, 100 policies that consistently performed well from the set of generated policies were selected