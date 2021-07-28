'''
Run an individual simulation from the config.ini parameters 
and display and/or record the results.
'''

from configparser import ConfigParser

import cfmm
import time_series
from utils import getRiskyReservesGivenSpotPrice
from utils import getRisklessGivenRisky
from arb import arbitrageExactly
from simulate import simulate

import matplotlib.pyplot as plt 
import numpy as np
import scipy
from scipy.stats import norm

import math

EPSILON = 1e-8

#Import config 
config_object = ConfigParser()
config_object.read("config.ini")

STRIKE_PRICE = float(config_object.get("Pool parameters", "STRIKE_PRICE"))
TIME_TO_MATURITY = float(config_object.get("Pool parameters", "TIME_TO_MATURITY"))
FEE = float(config_object.get("Pool parameters", "FEE"))

INITIAL_REFERENCE_PRICE = float(config_object.get("Price action parameters", "INITIAL_REFERENCE_PRICE"))
ANNUALIZED_VOL = float(config_object.get("Price action parameters", "ANNUALIZED_VOL"))
DRIFT = float(config_object.get("Price action parameters", "DRIFT"))
TIME_HORIZON = float(config_object.get("Price action parameters", "TIME_HORIZON"))
TIME_STEPS_SIZE = float(config_object.get("Price action parameters", "TIME_STEPS_SIZE"))

TAU_UPDATE_FREQUENCY = float(config_object.get("Simulation parameters", "TAU_UPDATE_FREQUENCY"))
SIMULATION_CUTOFF = float(config_object.get("Simulation parameters", "SIMULATION_CUTOFF"))
SEED = int(config_object.get("Simulation parameters", "SEED"))

IS_CONSTANT_PRICE = config_object.getboolean("Simulation parameters", "IS_CONSTANT_PRICE")
PLOT_PRICE_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PRICE_EVOL")
PLOT_PAYOFF_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PAYOFF_EVOL")
PLOT_PAYOFF_DRIFT = config_object.getboolean("Simulation parameters", "PLOT_PAYOFF_DRIFT")
SAVE_PRICE_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PRICE_EVOL")
SAVE_PAYOFF_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PAYOFF_EVOL")
SAVE_PAYOFF_DRIFT = config_object.getboolean("Simulation parameters", "SAVE_PAYOFF_DRIFT")

#Initialize pool parameters
sigma = ANNUALIZED_VOL
initial_tau = TIME_TO_MATURITY
K = STRIKE_PRICE
fee = FEE
gamma = 1 - FEE
np.random.seed(SEED)

#Stringify for plotting
gamma_str = str(1 - fee)
sigma_str = str(sigma)
K_str = str(K)

#Initialize pool and arbitrager objects
Pool = cfmm.CoveredCallAMM(0.5, K, sigma, initial_tau, fee)

#Initialize GBM parameters
T = TIME_HORIZON
dt = TIME_STEPS_SIZE
S0 = INITIAL_REFERENCE_PRICE


t, S = time_series.generateGBM(T, DRIFT, ANNUALIZED_VOL, S0, dt)

if IS_CONSTANT_PRICE:
    length = len(S)
    constant_price = []
    for i in range(length):
        constant_price.append(S0)
    S = constant_price

plt.plot(t, S)
plt.show()

# Prepare storage variables

# Store spot prices after each step
spot_price_array = []
# Marginal price affter each step
min_marginal_price_array = []
max_marginal_price_array = []

# Array to store the theoretical value of LP shares in the case of a pool with zero fees
theoretical_lp_value_array = []
# Effective value of LP shares with fees
effective_lp_value_array = []

dtau = TAU_UPDATE_FREQUENCY

for i in range(len(S)):
    # if i % 36 == 0: 
    #     print("In progress... ", round(i/365), "%")
    #Update pool's time to maturity
    theoretical_tau = initial_tau - t[i]

    print(f"___________________ \n \nStep {i}")
    print("Invariant before updating tau =  ", Pool.invariant)
    
    if i % dtau == 0:
        # print("hey")
        Pool.tau = initial_tau - t[i]
        print(f"New value of tau: {Pool.tau}")
        #Changing tau changes the value of the invariant even if no trade happens
        Pool.invariant = Pool.reserves_riskless - Pool.getRisklessGivenRiskyNoInvariant(Pool.reserves_risky)
        print("Invariant after updating tau =  ", Pool.invariant)
        spot_price_array.append(Pool.getSpotPrice())
        # _, max_marginal_price = Pool.virtualSwapAmountInRiskless(EPSILON)
        # _, min_marginal_price = Pool.virtualSwapAmountInRisky(EPSILON)
    
    # This is to avoid numerical errors that have been observed when getting 
    # closer to maturity. TODO: Figure out what causes these numerical errors.

    if Pool.tau >= 0:
        #Perform arbitrage step
        arbitrageExactly(S[i], Pool)
        max_marginal_price_array.append(Pool.getMarginalPriceSwapRisklessIn(0))
        min_marginal_price_array.append(Pool.getMarginalPriceSwapRiskyIn(0))
        #Get reserves given the reference price in the zero fees case
        theoretical_reserves_risky = getRiskyReservesGivenSpotPrice(S[i], Pool.K, Pool.sigma, theoretical_tau)
        theoretical_reserves_riskless = getRisklessGivenRisky(theoretical_reserves_risky, Pool.K, Pool.sigma, theoretical_tau)
        theoretical_lp_value = theoretical_reserves_risky*S[i] + theoretical_reserves_riskless
        theoretical_lp_value_array.append(theoretical_lp_value)
        effective_lp_value_array.append(Pool.reserves_risky*S[i] + Pool.reserves_riskless)
        # max_marginal_price_array.append(max_marginal_price)
        # min_marginal_price_array.append(min_marginal_price)
    if Pool.tau < 0: 
        max_index = i
        break
    max_index = i

# plt.plot(fees, mse, 'o')
# plt.xlabel("Fee")
# plt.ylabel("MSE")
# plt.title("Mean square error with theoretical payoff as a function of the fee parameter\n" + r"$\sigma = 0.5$, $K = 1100$, $\gamma = 1$, $\mathrm{d}\tau = 30 \ \mathrm{days}$")
# plt.show()

theoretical_lp_value_array = np.array(theoretical_lp_value_array)
effective_lp_value_array = np.array(effective_lp_value_array)

#Mean square error
mse = np.square(np.subtract(theoretical_lp_value_array, effective_lp_value_array)/theoretical_lp_value_array).mean()

# Since we're adding the marginal prices to the array at the 
# nex ste, after the tau and k update, in order to 
#  to avoid representing diverging prices in the zero
# fee case, we need to offset S so that the plot correctly shows
# the marginal prices in the pool *after* arbitrage and 
# *after* an update in tau and corresponding invariant when arbitrage 
# occurs.
S_offset = []

for i in range(len(S)-1):
    S_offset.append(S[i+1])

# Note: as a result, we're plotting the price after arbitrage, and after the 
# NEW update in tau, so the prices display actually differ. 

if PLOT_PRICE_EVOL: 
    plt.plot(t[0:max_index], S[0:max_index], label = "Reference price")
    # plt.plot(t[0:max_index], spot_price_array, label = "Pool spot price")
    plt.plot(t[0:max_index], min_marginal_price_array[0:max_index], label = "Price sell risky")
    plt.plot(t[0:max_index], max_marginal_price_array[0:max_index], label = "Price buy risky")
    plt.title("Arbitrage between CFMM and reference price\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=ANNUALIZED_VOL, strike=STRIKE_PRICE, gam=1-FEE, dt=TIME_STEPS_SIZE)+" days"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Price (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+str(ANNUALIZED_VOL)+"_K"+str(STRIKE_PRICE)+"_gamma"+str(gamma)+"_dtau"+str(TIME_STEPS_SIZE)+"_seed"+str(SEED)
    filename = 'price_evol_'+params_string+'.svg'
    plt.plot()
    if SAVE_PRICE_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = False)

if PLOT_PAYOFF_EVOL:
    plt.figure()
    plt.plot(t[0:max_index], theoretical_lp_value_array[0:max_index], label = "Theoretical LP value")
    plt.plot(t[0:max_index], effective_lp_value_array[0:max_index], label = "Effective LP value")
    plt.title("Value of LP shares\n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=ANNUALIZED_VOL, strike=STRIKE_PRICE, gam=1-FEE, dt=TIME_STEPS_SIZE)+" days"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Value (USD)")
    plt.legend(loc='best')
    params_string = "sigma"+str(ANNUALIZED_VOL)+"_K"+str(STRIKE_PRICE)+"_gamma"+str(gamma)+"_dtau"+str(TAU_UPDATE_FREQUENCY)+"_seed"+str(SEED)
    filename = 'lp_value_'+params_string+'.svg'
    plt.plot()
    if SAVE_PAYOFF_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = True)


if PLOT_PAYOFF_DRIFT:
    plt.figure()
    plt.plot(t[0:max_index], 100*abs(theoretical_lp_value_array[max_index]-effective_lp_value_array[max_index])/theoretical_lp_value_array, label=f"Seed = {SEED}")
    plt.title("Drift of LP shares value vs. theoretical \n" + r"$\sigma = {vol}$, $K = {strike}$, $\gamma = {gam}$, $d\tau = {dt}$".format(vol=ANNUALIZED_VOL, strike=STRIKE_PRICE, gam=1-FEE, dt=TIME_STEPS_SIZE)+" days"+ ", np.seed("+str(SEED)+")")
    plt.xlabel("Time steps (days)")
    plt.ylabel("Drift (%)")
    plt.legend(loc='best')
    params_string = "sigma"+str(ANNUALIZED_VOL)+"_K"+str(STRIKE_PRICE)+"_gamma"+str(gamma)+"_dtau"+str(TAU_UPDATE_FREQUENCY)+"_seed"+str(SEED)
    filename = 'drift_seed_comparison'+params_string+'.svg'
    plt.plot()
    if SAVE_PAYOFF_DRIFT:
        plt.savefig('sim_results/'+filename)
    plt.show()

print("MSE = ", mse)
# print("final divergence = ", 100*abs(theoretical_lp_value_array[-1] - effective_lp_value_array[-1])/theoretical_lp_value_array[-1], "%")