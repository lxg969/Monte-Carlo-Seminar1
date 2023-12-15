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
from math import factorial

sympy.init_printing()

class Pricer:
    def __init__(self, S0, r, sigma, T, K, paths, I, LNparams, JRparams, GBMparams):
        self.S0 = S0
        self.r = r
        self.T = T
        self.K = K
        self.paths = paths
        self.I = I
        self.sigma = sigma
        self.LNparams = LNparams
        self.JRparams = JRparams
        self.GBMparams = GBMparams
        

    def closed_formula_GBM(self,S0, K, T, r, sigma):
        d1 = (np.log(S0 / K) + (r + sigma**2 / 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        return K * np.exp(-r * T) * ss.norm.cdf(-d2) - S0 * ss.norm.cdf(-d1)
    
    def closed_formula_LN(self, S0, K, T, r, jumps):
        tot = 0
        LN_lam, LN_sigma, LN_mu, LN_v, m = self.LNparams
        GBM_sigma, GBM_mu = self.GBMparams
        for n in range(jumps):
            tot += (np.exp(-LN_lam * T) * (LN_lam * T) ** n / factorial(n))*self.closed_formula_GBM(S0, K, T, r, np.sqrt(GBM_sigma**2 + n * LN_v**2 / T))
            return tot
        
    def closed_formula_JR(self, S0, K, T, rente, sigma):
        return self.closed_formula_GBM(S0, K, T, rente, sigma)
    
    def crude_monte_carlo_european(self, paths):
        price = np.maximum(self.K - paths[:, -1], 0)  # payoff at maturity
        price *= np.exp(-self.r * self.T)
        return price


    #def closed_formula_LN(self):
    #    LN_lam, LN_sigma, LN_mu, LN_v, m = self.LNparams
    #    m = LN_lam * (np.exp(LN_mu + (LN_sigma**2) / 2) - 1)
    #    lam2 = LN_lam * np.exp(LN_mu + LN_sigma**2) / 2
    #    v = LN_v
#
    #    tot = 0
    #    for k in range(3):
    #        tot += (np.exp(-lam2 * self.T) * (lam2 * self.T) ** k / np.math.factorial(k)) * self.closed_formula_GBM(
    #        LN_mu - m + k * (LN_mu + 0.5 * LN_sigma**2) / self.T,
    #        np.sqrt(LN_sigma**2 + k * v**2 / self.T),
    #        )
    #        return tot

    #def closed_formula_Call_JR(self):
    #    JR_lam, JR_sigma, JR_mu, JR_v, m = self.JRparams = self.JRparams
    #    return np.exp(-JR_lam * self.T) * self.closed_formula_GBM_call(self.r, 0.04)
#
    #def closed_formula_Put_JR(self):
     #   return self.closed_formula_Call_JR() - self.S0 + self.K * np.exp(-self.r * self.T)

    def update_exercise_matrix(self, exercise_matrix):
        for path in range(exercise_matrix.shape[0]):
            first_exercise = np.where(exercise_matrix[path])[0]
            if first_exercise.size > 0:
                # Set all values after the first exercise to False
                exercise_matrix[path, first_exercise[0] + 1:] = False
        return exercise_matrix

    def LSM(self, S, basis, deg):
        df = np.exp(-self.r * self.T/self.I)
        paths = len(S)
        H = np.maximum(self.K - S, 0)  # intrinsic values for put option
        V = np.zeros_like(H)  # value matrix
        V[:, -1] = H[:, -1]  # set value at maturity equal to intrinsic value
        exercise_matrix = np.zeros_like(H, dtype=bool)  # matrix to track exercise decisions

        # Valuation by LS Method
        for t in range(self.I - 2, 0, -1):
            good_paths = H[:, t] > 0  # paths where the intrinsic value is positive

            if np.sum(good_paths) > 0:
                if basis == 'poly':
                    rg = np.polyfit(S[good_paths, t], V[good_paths, t + 1] * df, deg)
                    C = np.polyval(rg, S[good_paths, t])
                elif basis == 'legendre':
                    rg = np.polynomial.legendre.legfit(S[good_paths, t], V[good_paths, t + 1] * df, deg)
                    C = np.polynomial.legendre.legval(S[good_paths, t], rg)
                elif basis =='laguerre':
                    rg = np.polynomial.laguerre.lagfit(S[good_paths, t], V[good_paths, t + 1] * df, deg)
                    C = np.polynomial.laguerre.lagval(S[good_paths, t], rg)
                else:  # 'hermite'
                    rg = np.polynomial.hermite.hermfit(S[good_paths, t], V[good_paths, t + 1] * df, deg)
                    C = np.polynomial.hermite.hermval(S[good_paths, t], rg)

                exercise = H[good_paths, t] > C
                exercise_full = np.zeros_like(H[:, t], dtype=bool)
                exercise_full[good_paths] = exercise
            else:
                exercise_full = np.zeros_like(H[:, t], dtype=bool)

            exercise_matrix[:, t] = exercise_full
            V[exercise_full, t] = H[exercise_full, t]
            V[exercise_full, t + 1:] = 0
            discount_path = ~exercise_full
            V[discount_path, t] = V[discount_path, t + 1] * df

        exercise_matrix = self.update_exercise_matrix(exercise_matrix)

        V0 = np.mean(V[:, 1]) * df  # discounted expectation of V[t=1]
        V0_array = V[:, 1] * df
        SE = np.std(V[:, 1] * df) / np.sqrt(paths)
        return V0, V0_array, SE, exercise_matrix, V