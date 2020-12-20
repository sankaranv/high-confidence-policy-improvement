# High Confidence Policy Improvement

Course Project for COMPSCI 687: Reinforcement Learning

# Requirements

Python 3.7 or higher, numpy, scipy, cma, tqdm, pandas

# Instructions

All commands are run from the root directory

- To generate the dataset, place data.csv in the root directory and run proc_data.py
This will output to dataset.pkl (the file will be over 2GB)

- To generate policies, run hcope.py
Policies will be stored in {root-dir}/policies/

- To test the generated policies, run eval_safety.py
Tested policies will be stored in {root-dir}/policies/checked
For the final submission, 100 policies that consistently performed well from the set of generated policies were selected

# References

P. S. Thomas, G. Theocharous, and M. Ghavamzadeh. High Confidence Policy Improvement. In Proceedings of the Thirty-Second International Conference on Machine Learning, 2015

P. S. Thomas, B. Castro da Silva, A. G. Barto, S. Giguere, Y. Brun, and E. Brunskill. Preventing undesirable behavior of intelligent machines. Science vol. 366, Issue 6468, pages 999–1004, 2019