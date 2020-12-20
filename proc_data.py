import csv
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import pickle

num_trajectories = 1000000
dataset = defaultdict()
pbar = tqdm(total=num_trajectories)
idx = -1
time = 0
print("Reading dataset from file...")
with open('data.csv', newline='') as csvfile:
    reader = csv.reader(csvfile)
    for i,row in enumerate(csvfile):
        if i == 0:
            continue
        row = list(map(float, row.rstrip().split(',')))
        if len(row) == 1:
            pbar.update(1)
            idx += 1
            time = 0
            dataset[idx] = []
        elif len(row) == 4:
            # dataset[idx].append([idx] + [time] + row)
            dataset[idx].append(row)
            time += 1
        else:
            raise ValueError("Error in row {}".format(i))

print("Dataset imported from file! Converting to numpy...")
pbar = tqdm(total=num_trajectories)
for i in range(num_trajectories):
    pbar.update(1)
    dataset[i] = np.array(dataset[i])
    
print("Converted to numpy! Saving dataset...")  
with open('dataset.pkl', 'wb') as handle:
    pickle.dump(dataset, handle, protocol=pickle.HIGHEST_PROTOCOL)
print("Dataset saved")

