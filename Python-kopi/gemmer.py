import matplotlib.pyplot as plt
plt.style.use('ggplot')
import numpy as np
from numpy import zeros_like
from IPython.display import display
def display_matrix(m):
    display(sympy.Matrix(m))
import sympy
import pandas as pd
import scipy.stats as ss

sympy.init_printing()

from Pricer import Pricer
from Processes import Processes
from Processes_combined import Processes1

from scipy.stats.qmc import Sobol
from scipy.stats import norm

def gbm_paths_antithetic_moment_matching(self):
    mu, sigma = self.GBMparams
    S0, paths, I, T = self.S0, self.paths, self.I, self.T
    dt = T / I
    matrix = np.zeros((2 * self.paths, I))  # Double the number of paths for antithetic paths
    all_normals = []  # Store all normal variables for moment matching
    # First, generate original paths and store the normal variables
    for k in range(self.paths):
        S = np.zeros(I)
        S[0] = S0
        for i in range(1, I):
            Z = np.random.standard_normal()
            all_normals.append(Z)
            S[i] = S[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
        matrix[k] = S
    # Calculate mean and std of normal variables
    mean_normal = np.mean(all_normals)
    std_normal = np.std(all_normals)
    # Now generate antithetic paths using moment matching
    for k in range(self.paths):
        S_anti = np.zeros(I)
        S_anti[0] = S0
        for i in range(1, I):
            Z = -all_normals[k * (I - 1) + i - 1]  # Use stored normal variable and negate it
            adjusted_Z = (Z - mean_normal) / std_normal  # Adjust Z using moment matching
            S_anti[i] = S_anti[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * adjusted_Z)
        matrix[self.paths + k] = S_anti
    return matrix

def merton_jump_paths_antithetic_moment_matching(self):
    lam, sigma, mu, v, m = self.LNparams
    S0, paths, I, T = self.S0, self.paths, self.I, self.T
    dt = T / I
    matrix = np.zeros((2 * paths, I))
    all_normals, all_lognormals = [], []
    # Generate original paths and store normal and lognormal variables
    for k in range(paths):
        X = np.zeros(I)
        X[0] = np.log(S0)
        S = np.zeros(I)
        S[0] = S0
        for i in range(1, I):
            Z = np.random.standard_normal()
            all_normals.append(Z)
            N = np.random.poisson(lam * dt)
            if N > 0:
                Y = np.exp(np.random.normal(m, v, N))
                logY = np.log(Y)
                all_lognormals.extend(logY)
                M = np.sum(logY)
            else:
                M = 0
            X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z + M
            S[i] = np.exp(X[i])
        matrix[k] = S
    # Calculate mean and std for normal and lognormal variables
    mean_normal = np.mean(all_normals)
    std_normal = np.std(all_normals)
    mean_lognormal = np.mean(all_lognormals) if all_lognormals else 0
    std_lognormal = np.std(all_lognormals) if all_lognormals else 0
    # Generate antithetic paths using moment matching
    for k in range(paths):
        X_anti = np.zeros(I)
        X_anti[0] = np.log(S0)
        S_anti = np.zeros(I)
        S_anti[0] = S0
        for i in range(1, I):
            Z = -all_normals[k * (I - 1) + i - 1]
            Z_adjusted = (Z - mean_normal) / std_normal
            N = np.random.poisson(lam * dt)
            if N > 0:
                Y_anti = np.exp(np.random.normal(m, v, N))
                logY_anti = np.log(Y_anti)
                M_anti = np.sum((logY_anti - mean_lognormal) / std_lognormal * v + m)
            else:
                M_anti = 0
            X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_adjusted + M_anti
            S_anti[i] = np.exp(X_anti[i])
        matrix[paths + k] = S_anti
    return matrix
def merton_jump_to_ruin_paths_antithetic_moment_matching(self):
    lam, sigma, mu = self.JRparams
    S0, paths, I, T = self.S0, self.paths, self.I, self.T
    dt = T / I
    matrix = np.zeros((2 * paths, I))
    all_normals = []
    # Generate original paths and store normal variables
    for k in range(paths):
        X = np.zeros(I)
        X[0] = np.log(S0)
        S = np.zeros(I)
        S[0] = S0
        for i in range(1, I):
            Z = np.random.standard_normal()
            all_normals.append(Z)
            N = np.random.poisson(lam * dt)
            if N == 0 and X[i-1] > 0:
                X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
                S[i] = np.exp(X[i])
            else:
                S[i] = 0
        matrix[k] = S
    # Calculate mean and std for normal variables
    mean_normal = np.mean(all_normals)
    std_normal = np.std(all_normals)
    # Generate antithetic paths using moment matching
    for k in range(paths):
        X_anti = np.zeros(I)
        X_anti[0] = np.log(S0)
        S_anti = np.zeros(I)
        S_anti[0] = S0
        for i in range(1, I):
            Z = -all_normals[k * (I - 1) + i - 1]
            Z_adjusted = (Z - mean_normal) / std_normal
            N = np.random.poisson(lam * dt)
            if N == 0 and X_anti[i-1] > 0:
                X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_adjusted
                S_anti[i] = np.exp(X_anti[i])
            else:
                S_anti[i] = 0
        matrix[paths + k] = S_anti
    return matrix