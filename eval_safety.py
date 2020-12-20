import csv
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from collections import defaultdict
import pickle
import cma
from scipy import stats
import sys
import time
import os

class HCOPE:

	def __init__(self):
		self.num_states = 18
		self.num_actions = 4
		self.num_episodes = 1000000
		self.maxiter = 25
		self.candidate_split = 0.05
		self.gamma = 0.95
		self.sigma = 1
		self.delta = 0.01
		self.baseline = 1.41537 * 1.1

		self.load_data()

	def load_data(self, path = 'dataset.pkl'):
		with open(path, 'rb') as handle:
			self.dataset = pickle.load(handle)
			self.num_candidate_episodes = int(self.candidate_split * self.num_episodes)
			self.num_safety_episodes = int((1 - self.candidate_split) * self.num_episodes)
			indices = np.arange(self.num_episodes)
			np.random.shuffle(indices)
			self.candidate_indices = indices[:self.num_candidate_episodes]
			self.safety_indices = indices[self.num_candidate_episodes:]


	def get_softmax_probs(self, theta):
		policy = np.copy(theta).reshape(self.num_states,self.num_actions)
		for s in range(self.num_states):
			policy[s] = np.exp(policy[s])
			policy[s] = policy[s] / np.sum(policy[s])
		return policy

	def compute_pdis_candidate(self, theta_e):
		pi_e = self.get_softmax_probs(theta_e)
		pdis_estimates = []
		for t in self.candidate_indices:
			horizon = len(self.dataset[t])
			probs_b = self.dataset[t][:,-1]
			states = np.array(self.dataset[t][:,0]).astype(int)
			actions = np.array(self.dataset[t][:,1]).astype(int)
			rewards = np.array(self.dataset[t][:,2])
			probs_e = pi_e[states,actions]
			importance_ratio = np.exp(np.cumsum(np.log(probs_e)) - np.cumsum(np.log(probs_b)))
			gamma_vec = np.power(self.gamma, np.arange(horizon))
			pdis_estimates.append(np.sum(gamma_vec * importance_ratio * rewards))
		pdis_mean = np.mean(pdis_estimates)
		pdis_stddev = np.sqrt((1./(self.num_candidate_episodes-1) * np.sum((pdis_estimates - pdis_mean)**2))) 
		return pdis_mean, pdis_stddev

	def compute_pdis_safety(self, theta_e):
		pi_e = self.get_softmax_probs(theta_e)
		pdis_estimates = []
		for t in self.safety_indices:
			horizon = len(self.dataset[t])
			probs_b = self.dataset[t][:,-1]
			states = np.array(self.dataset[t][:,0]).astype(int)
			actions = np.array(self.dataset[t][:,1]).astype(int)
			rewards = np.array(self.dataset[t][:,2])
			probs_e = pi_e[states,actions]
			importance_ratio = np.exp(np.cumsum(np.log(probs_e)) - np.cumsum(np.log(probs_b)))
			gamma_vec = np.power(self.gamma, np.arange(horizon))
			pdis_estimates.append(np.sum(gamma_vec * importance_ratio * rewards))
		pdis_mean = np.mean(pdis_estimates)
		pdis_stddev = np.sqrt((1./(self.num_safety_episodes-1) * np.sum((pdis_estimates - pdis_mean)**2))) 
		return pdis_mean, pdis_stddev

	def objective(self,thetas):
		obj = []
		for theta in thetas:
			pdis, stddev = self.compute_pdis_candidate(theta)
			constraint = self.constraint(theta)
			if constraint >= 0:
				obj.append(-pdis)
			else:
				obj.append(10e6 - constraint)		
		return obj

	def constraint(self,theta):
		pdis, stddev = self.compute_pdis_candidate(theta)
		t_test_value = stats.t.ppf(1 - self.delta, self.num_safety_episodes - 1)
		constraint = pdis - ((2 * stddev / np.sqrt(self.num_safety_episodes)) * t_test_value) - self.baseline
		return constraint


	def safety_test(self, theta, filename):
		pdis, stddev = self.compute_pdis_safety(theta)
		t_test_value = stats.t.ppf(1 - self.delta, self.num_safety_episodes - 1)
		constraint = pdis - (stddev / np.sqrt(self.num_safety_episodes)) * t_test_value - self.baseline
		safe = False
		if constraint >= self.baseline:
			safe = True
		print("{}: Safe policy bound: {} vs. baseline {} => {}".format(filename, constraint, self.baseline, safe))
		return constraint, safe

	def get_candidate_policy(self):
		theta = np.random.normal(loc=0, scale=1, size=(self.num_states,self.num_actions)).reshape(-1)
		es = cma.CMAEvolutionStrategy(theta,self.sigma,{'popsize':5})
		i = 0
		while((not es.stop()) and i < self.maxiter):
			soln = es.ask()
			es.tell(soln, self.objective(soln))
			es.logger.add()
			es.disp()
			i+=1
		cand_score = self.constraint(es.result[0]) + self.baseline
		print("Candidate policy bound: {} vs. baseline {}".format(cand_score, self.baseline))
		return es.result[0], i

	def run(self):
		self.theta, iters = self.get_candidate_policy()
		if iters < 20:
			print("Failed Candidate Safety Test")
			return -200, False
			
		cand_score = self.constraint(self.theta)
		if cand_score < 0:
			print("Failed Candidate Safety Test")
			return -200, False
		else:
			self.policy = self.get_softmax_probs(self.theta)
			constraint, safe = self.safety_test(self.theta)
			if safe:
				print("Passed Safety Test")
				return constraint, True
			else:
				print("Failed Safety Test")
				return constraint, False
	
def main():

	succ = {}
	for file in os.scandir("policies"):
		if file.path.endswith(".txt"):
			succ[file.path] = 0
			
	num_tests = 5

	for t in range(num_tests):

		hcope = HCOPE()
		hcope.delta = 0.01
		hcope.candidate_split = 0.05
		hcope.baseline = 1.41537 * 1.1

		print("Test {}".format(t+1))
		for file in os.scandir("policies"):
			if file.path.endswith(".txt"):
				theta = np.loadtxt(file.path)
				_, safe = hcope.safety_test(theta, file.path)
				if safe:
					succ[file.path] += 1

	file_idx = 1                
	for file, score in succ.items():
		if score == num_tests:
			theta = np.loadtxt(file)
			np.savetxt('policies/checked/policy{}.txt'.format(file_idx, theta), theta)
			file_idx += 1

if __name__ == "__main__":
	main()