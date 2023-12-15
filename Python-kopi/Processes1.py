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
from scipy.stats.qmc import Sobol
from scipy.stats import norm
from scipy.stats import qmc, norm

sympy.init_printing()

class Processes1():
    def __init__(self, S0, r, T, K, paths, I, LNparams, JRparams, GBMparams):
        print ("Processes1 class created with S0 = %s, paths = %s, I = %s, T = %s" % (S0, paths, I, T))
        self.S0 = S0
        self.r = r
        self.T = T
        self.K = K
        self.paths = paths
        self.I = I
        self.LNparams = LNparams
        self.JRparams = JRparams
        self.GBMparams = GBMparams

    def sobol_norm(self, d=1):
        I, paths = self.I, self.paths
        sampler = qmc.Sobol(d, scramble=True)
        x_sobol = sampler.random_base2(m=int(np.log2(I*paths)))
        np.random.shuffle(x_sobol)
        return ss.norm.ppf(x_sobol)

    def gbm_paths_sobol_antithetic(self):
        mu, sigma = self.GBMparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((2 * self.paths, I))  # Double the number of paths

        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0

        for k in range(self.paths):
            S = np.zeros(I)
            S_anti = np.zeros(I)  # For antithetic path

            S[0] = S0
            S_anti[0] = S0

            for i in range(1, I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                Z_anti = -Z

                #original path
                S[i] = S[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

                #antithetic path
                S_anti[i] = S_anti[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_anti)

            matrix[k] = S
            matrix[self.paths + k] = S_anti
        return matrix

    
    def merton_jump_paths_sobol_antithetic(self):
        lam , sigma, mu, v, m = self.LNparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((2 * paths, I))  # Twice the number of paths for antithetic paths
        dt = T / I
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0

        for k in range(paths):
            X = np.zeros(I)
            X_anti = np.zeros(I)  # For antithetic path
            S = np.zeros(I)
            S_anti = np.zeros(I)  # For antithetic path

            S[0], S_anti[0] = S0, S0
            X[0], X_anti[0] = np.log(S0), np.log(S0)

            for i in range(1, I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                Z_anti = -Z
                N = np.random.poisson(lam * dt)
                Y = np.exp(np.random.normal(m, v, N))

                M = np.sum(np.log(Y)) if N > 0 else 0
                M_anti = -M  # Negating the jump component for antithetic path

                # Original path
                X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z + M
                S[i] = np.exp(X[i])

                # Antithetic path
                X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_anti + M_anti
                S_anti[i] = np.exp(X_anti[i])

            matrix[k] = S
            matrix[paths + k] = S_anti  # Storing the antithetic path

        return matrix
    
    def merton_jump_to_ruin_paths_sobol_antithetic(self):
        lam , sigma, mu = self.JRparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((2 * paths, I))
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0  # Index to track the current quasi-random number
        dt = T / I
        for k in range(paths):
            X = np.zeros(I)
            X_anti = np.zeros(I)

            S = np.zeros(I)
            S_anti = np.zeros(I)

            X[0], X_anti[0] = np.log(S0), np.log(S0)
            S[0], S_anti[0] = S0, S0

            

            for i in range(1, I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                Z_anti = -Z
                N = np.random.poisson(lam * dt)

                # Original path
                if N == 0 and X[i-1] > 0:
                    X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
                    S[i] = np.exp(X[i])
                else:
                    S[i] = 0

                # Antithetic path
                if N == 0 and X_anti[i-1] > 0:
                    X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_anti
                    S_anti[i] = np.exp(X_anti[i])
                else:
                    S_anti[i] = 0

            matrix[k] = S
            matrix[paths + k] = S_anti
        return matrix
    

    def gbm_paths_sample(self):
        mu, sigma = self.GBMparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((paths, I))
        save_normal = np.zeros((paths, I))
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0
        for k in range(paths):
            S = np.zeros(I)
            S[0] = S0
            for i in range(1, I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                S[i] = S[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            matrix[k] = S
            save_normal[k] = Z
        mean_normal = np.mean(save_normal)
        std_normal = np.std(save_normal)
        return mean_normal, std_normal
    
    def gbm_paths_sobol_moment_matching(self, mean_normal, std_normal):
        mu, sigma = self.GBMparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((paths, I))
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0

        for k in range(paths):
            S = np.zeros(I)
            S[0] = S0
            for i in range(1, I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                S[i] = S[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z-mean_normal)/std_normal))
            matrix[k] = S
        return matrix
            
    def merton_jump_to_ruin_paths_sample(self):
        lam , sigma, mu = self.JRparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((paths, I))
        dt = T / I
        all_normals = []
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0
        for k in range(paths):
            X = np.zeros(I)
            S = np.zeros(I)
            X[0] = np.log(S0)
            S[0] = S0

            for i in range(1,I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                all_normals.append(Z)
                N = np.random.poisson(lam * dt)
                if N == 0 and X[i-1] > 0:
                    X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
                    S[i] = np.exp(X[i])
                else:
                    S[i] = 0
            matrix[k] = S
        mean_normal = np.mean(all_normals)
        std_normal = np.std(all_normals)
        return mean_normal, std_normal

    def merton_jump_to_ruin_paths_sobol_moment_matching(self, mean_normal_JR, std_normal_JR):
        lam , sigma, mu = self.JRparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((paths, I))
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0
        dt = T / I
        for k in range(paths):
            X = np.zeros(I)
            S = np.zeros(I)
            X[0] = np.log(S0)
            S[0] = S0
            for i in range(1,I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                N = np.random.poisson(lam * dt)
                if N == 0 and X[i-1] > 0:
                    X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * (Z - mean_normal_JR) / std_normal_JR
                    S[i] = np.exp(X[i])
                else:
                    S[i] = 0
            matrix[k] = S
        return matrix
    
    def merton_jump_paths_sample(self):
        lam, sigma, mu, v, m = self.LNparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((paths, I))
        all_normals = []
        all_lognormals = []
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0

        for k in range(paths):
            X = np.zeros(I)
            S = np.zeros(I)
            S[0] = S0
            X[0] = np.log(S0)

            for i in range(1, I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                all_normals.append(Z)

                N = np.random.poisson(lam * dt)
                if N > 0:
                    Y = np.exp(np.random.normal(m, v, N))
                    logY = np.log(Y)
                    M = np.sum(np.log(Y))
                    all_lognormals.extend(logY)
                else:
                    M = 0

                X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z + M
                S[i] = np.exp(X[i])
            matrix[k] = S

        mean_normal = np.mean(all_normals)
        std_normal = np.std(all_normals)

        if all_lognormals:
            mean_lognormal = np.mean(all_lognormals)
            std_lognormal = np.std(all_lognormals)
        else:
            mean_lognormal = 0
            std_lognormal = 0

        return mean_normal, mean_lognormal, std_normal, std_lognormal
    
    def merton_jump_paths_sobol_moment_matching(self, mean_normal_LN, mean_lognormal_LN, std_normal_LN, std_lognormal_LN):
        lam, sigma, mu, v, m = self.LNparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((paths, I))
        quasi_random_numbers = self.sobol_norm()
        quasi_random_index = 0
        for k in range(paths):
            X = np.zeros(I)
            S = np.zeros(I)
            S[0] = S0
            X[0] = np.log(S0)
            for i in range(1,I):
                Z = quasi_random_numbers[quasi_random_index % len(quasi_random_numbers)]
                quasi_random_index += 1
                N = np.random.poisson(lam * dt)
                Y = np.exp(np.random.normal(m,v,N))
                logy = np.log(Y)

                if N == 0:
                    M = 0
                else:
                    M = np.sum(((logy - mean_lognormal_LN)/std_lognormal_LN)*v+m)
                X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z-mean_normal_LN)/std_normal_LN) + M
                S[i] = np.exp(X[i])
            matrix[k] = S
        return matrix
    
    def gbm_paths_antithetic_sample(self):
        mu, sigma = self.GBMparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((2 * self.paths, I))  # Double the number of paths
        save_normal = np.zeros((2 * paths, I))
        for k in range(self.paths):
            S = np.zeros(I)
            S_anti = np.zeros(I)  # For antithetic path

            S[0] = S0
            S_anti[0] = S0

            for i in range(1, I):
                Z = np.random.standard_normal()
                Z_anti = -Z

                #original path
                S[i] = S[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)

                #antithetic path
                S_anti[i] = S_anti[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_anti)

            matrix[k] = S
            matrix[self.paths + k] = S_anti
            save_normal[k] = Z
        mean_normal = np.mean(save_normal)
        std_normal = np.std(save_normal)
        return mean_normal, std_normal
    
    def gbm_paths_antithetic_moment_matching(self, mean_normal, std_normal):
        mu, sigma = self.GBMparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        dt = T / I
        matrix = np.zeros((2 * self.paths, I))

        for k in range(self.paths):
            S = np.zeros(I)
            S_anti = np.zeros(I)

            S[0] = S0
            S_anti[0] = S0

            for i in range(1, I):
                Z = np.random.standard_normal()
                Z_anti = -Z

                #original path
                S[i] = S[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z-mean_normal)/std_normal))

                #antithetic path
                S_anti[i] = S_anti[i-1] * np.exp((mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z_anti-mean_normal)/std_normal))

            matrix[k] = S
            matrix[self.paths + k] = S_anti
        return matrix
    
    def merton_jump_paths_antithetic_sample(self):
        lam , sigma, mu, v, m = self.LNparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((2 * paths, I))
        all_normals = []
        all_lognormals = []

        dt = T / I

        for k in range(paths):
            X = np.zeros(I)
            X_anti = np.zeros(I)
            S = np.zeros(I)
            S_anti = np.zeros(I)

            S[0], S_anti[0] = S0, S0
            X[0], X_anti[0] = np.log(S0), np.log(S0)

            for i in range(1, I):
                Z = np.random.standard_normal()
                Z_anti = -Z
                all_normals.append(Z)

                N = np.random.poisson(lam * dt)
                if N > 0:
                    Y = np.exp(np.random.normal(m, v, N))
                    logY = np.log(Y)
                    M = np.sum(logY)
                    all_lognormals.extend(logY)
                else:
                    M = 0

                # Original path
                X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z + M
                S[i] = np.exp(X[i])

                # Antithetic path
                X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_anti + M
                S_anti[i] = np.exp(X_anti[i])

            matrix[k] = S
            matrix[paths + k] = S_anti

        mean_normal = np.mean(all_normals)
        std_normal = np.std(all_normals)

        if all_lognormals:
            mean_lognormal = np.mean(all_lognormals)
            std_lognormal = np.std(all_lognormals)
        else:
            mean_lognormal = 0
            std_lognormal = 0

        return mean_normal, mean_lognormal, std_normal, std_lognormal
    

    def merton_jump_paths_antithetic_moment_matching(self, mean_normal_LN, mean_lognormal_LN, std_normal_LN, std_lognormal_LN):
        lam, sigma, mu, v, m = self.LNparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((2 * paths, I))
        dt = T / I
        for k in range(paths):
            X = np.zeros(I)
            X_anti = np.zeros(I)
            S = np.zeros(I)
            S_anti = np.zeros(I)

            S[0], S_anti[0] = S0, S0
            X[0], X_anti[0] = np.log(S0), np.log(S0)

            for i in range(1, I):
                Z = np.random.standard_normal()
                Z_anti = -Z
                N = np.random.poisson(lam * dt)
                Y = np.exp(np.random.normal(m,v,N))
                logy = np.log(Y)

                if N == 0:
                    M = 0
                else:
                    M = np.sum(((logy - mean_lognormal_LN)/std_lognormal_LN)*v+m)
                # Original path
                X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z-mean_normal_LN)/std_normal_LN) + M
                S[i] = np.exp(X[i])

                # Antithetic path
                X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z_anti-mean_normal_LN)/std_normal_LN) + M
                S_anti[i] = np.exp(X_anti[i])

            matrix[k] = S
            matrix[paths + k] = S_anti

        return matrix
    
    def merton_jump_to_ruin_paths_antithetic_sample(self):
        lam , sigma, mu = self.JRparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((2 * paths, I))
        all_normals = []
        dt = T / I
        for k in range(paths):
            X = np.zeros(I)
            X_anti = np.zeros(I)
            S = np.zeros(I)
            S_anti = np.zeros(I)
            X[0], X_anti[0] = np.log(S0), np.log(S0)
            S[0], S_anti[0] = S0, S0
            for i in range(1,I):
                Z = np.random.standard_normal()
                Z_anti = -Z
                all_normals.append(Z)
                N = np.random.poisson(lam * dt)
                # Original path
                if N == 0 and X[i-1] > 0:
                    X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z
                    S[i] = np.exp(X[i])
                else:
                    S[i] = 0
                # Antithetic path
                if N == 0 and X_anti[i-1] > 0:
                    X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z_anti
                    S_anti[i] = np.exp(X_anti[i])
                else:
                    S_anti[i] = 0
            matrix[k] = S
            matrix[paths + k] = S_anti
        mean_normal = np.mean(all_normals)
        std_normal = np.std(all_normals)
        return mean_normal, std_normal
    
    def merton_jump_to_ruin_paths_antithetic_moment_matching(self, mean_normal_JR, std_normal_JR):
        lam , sigma, mu = self.JRparams
        S0, paths, I, T = self.S0, self.paths, self.I, self.T
        matrix = np.zeros((2 * paths, I))
        dt = T / I
        for k in range(paths):
            X = np.zeros(I)
            X_anti = np.zeros(I)
            S = np.zeros(I)
            S_anti = np.zeros(I)
            X[0], X_anti[0] = np.log(S0), np.log(S0)
            S[0], S_anti[0] = S0, S0
            for i in range(1,I):
                Z = np.random.standard_normal()
                Z_anti = -Z
                N = np.random.poisson(lam * dt)
                # Original path
                if N == 0 and X[i-1] > 0:
                    X[i] = X[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z-mean_normal_JR)/std_normal_JR)
                    S[i] = np.exp(X[i])
                else:
                    S[i] = 0
                # Antithetic path
                if N == 0 and X_anti[i-1] > 0:
                    X_anti[i] = X_anti[i-1] + (mu - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * ((Z_anti-mean_normal_JR)/std_normal_JR)
                    S_anti[i] = np.exp(X_anti[i])
                else:
                    S_anti[i] = 0
            matrix[k] = S
            matrix[paths + k] = S_anti
        return matrix