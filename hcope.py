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

class HCOPE:

	def __init__(self):
		self.num_states = 18
		self.num_actions = 4
		self.num_episodes = 1000000
		self.maxiter = 100
		self.candidate_split = 0.5
		self.gamma = 0.95
		self.sigma = 2.0
		self.delta = 0.05
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

	def objective(self,theta):
		pdis, stddev = self.compute_pdis_candidate(theta)
		if len(theta)!=18*4:
			print("Shape changed to {}".format(theta.shape))
		return -pdis

	def constraint(self,theta):
		start = time.time()
		pdis, stddev = self.compute_pdis_candidate(theta)
		t_test_value = stats.t.ppf(1 - self.delta, self.num_safety_episodes - 1)
		constraint = pdis - ((2 * stddev / np.sqrt(self.num_safety_episodes)) * t_test_value) - self.baseline
		return constraint


	def safety_test(self, theta):
		pdis, stddev = self.compute_pdis_safety(theta)
		t_test_value = stats.t.ppf(1 - self.delta, self.num_safety_episodes - 1)
		constraint = pdis - (stddev / np.sqrt(self.num_safety_episodes)) * t_test_value
		print("Safe policy bound: {} vs. baseline {}".format(constraint, self.baseline))

		if constraint >= self.baseline:
			return constraint, True
		else:
			return constraint, False

	def get_candidate_policy(self):
		theta = np.ones(18*4)
		argmin, es = cma.evolution_strategy.fmin_con(self.objective, theta, self.sigma, g=lambda x: [-self.constraint(x)], restart_from_best=True, options={'maxiter':self.maxiter})
		
		cand_score = self.constraint(argmin) + self.baseline
		print("Candidate policy bound: {} vs. baseline {}".format(cand_score, self.baseline))
		return argmin

	def run(self):
		self.theta = self.get_candidate_policy()
		self.policy = self.get_softmax_probs(self.theta)
		constraint, safe = self.safety_test(self.theta)
		if safe:
			print("Passed Safety Test")
			return constraint, True
		else:
			print("Failed Safety Test")
			return constraint, False
	
def main():
	num_safe_policies = 0
	target_num_safe_policies = 300
	runs = 0
	while num_safe_policies < target_num_safe_policies:
		safe = False
		constraint = 0
		hcope = HCOPE()
		while not safe:
			runs +=1
			print("-------------------------")
			print("Run {}".format(runs))
			constraint, safe = hcope.run()

		num_safe_policies += 1
		print("Found safe policy {} after {} runs".format(num_safe_policies,runs))
		np.savetxt('policies/policy{}.txt'.format(num_safe_policies),hcope.theta)
		cand_score = hcope.constraint(hcope.theta) + hcope.baseline
		np.savetxt('scores/score{}.txt'.format(num_safe_policies),np.array([cand_score, constraint]))

if __name__ == "__main__":
	main()


