"""
A collection of utility functions including some
covered call AMM relevant math.
"""

import math
from math import inf
import numpy as np
from scipy.optimize import newton
from scipy.stats import norm


def nonnegative(x):
    if isinstance(x, np.ndarray):
        return (x >= 0).all()
    return x >= 0



def getXGivenSpotPrice(S, K, sigma, tau):
    """
    WRITE THIS - we want to be able to get the reserves given some arbitrary spot price
    PUT IN THE CFMM FUNCTION
    """

    pass 




def getSellTrade(avg_size, variance, prob_X):
    """
    Generate a SELL trade of either X or Y with size drawn from a normal distribution

    Params:

    avg_size: Mean of the normal distriution 
    var: Variance of the normal distribution
    prob_X: probability trade is a SELL X

    """    

    trade_size = (int)np.random.normal(avg_size,variance)

    if np.random.rand() < prob_X:
        trade = 'SELL X'
    else:
        trade = 'SELL Y'

    return (trade, trade_size)


def generateGBM(T, mu, sigma, S0, dt):
    """
    Generate a geometric brownian motion time series. Shamelessly copy pasted from here: https://stackoverflow.com/a/13203189

    Params:

    T: time horizon
    mu: drift
    sigma: percentage volatility
    S0: initial price
    dt: size of time steps

    Returns:

    t: time array
    S: time series
    """
    N = round(T / dt)
    t = np.linspace(0, T, N)
    W = np.random.standard_normal(size=N)
    W = np.cumsum(W) * np.sqrt(dt)  ### standard brownian motion ###
    X = (mu - 0.5 * sigma ** 2) * t + sigma * W
    S = S0 * np.exp(X)  ### geometric brownian motion ###
    return t, S
