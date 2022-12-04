"""
Run an individual simulation from the config.ini parameters
and display and/or record the results.
"""

from configparser import ConfigParser

import matplotlib.pyplot as plt 
import numpy as np
import copy

from modules import constant_product_cfmm as cfmm
from modules.utils import getSellTrade, generateGBM



#Import config 
config_object = ConfigParser()
config_object.read("config.ini")


###### TO DO: Set reasonable parameters

FEE = float(config_object.get("Pool parameters", "FEE"))
EPSILON = float(config_object.get("Pool parameters", "EPSILON"))
K = float(config_object.get("Pool parameters", "K"))
BUDGET = float(config_object.get("Pool parameters", "BUDGET"))


TRADE_SIZE_AVERAGE = float(config_object.get("Trade parameters", "TRADE_SIZE_AVERAGE"))
TRADE_SIZE_VARIANCE  = float(config_object.get("Trade parameters", "TRADE_SIZE_VARIANCE"))
PROBABILITY_OF_TRADE_X = float(config_object.get("Trade parameters", "PROBABILITY_OF_TRADE_X"))

INITIAL_REFERENCE_PRICE = float(config_object.get("Price action parameters", "INITIAL_REFERENCE_PRICE"))
ANNUALIZED_VOL = float(config_object.get("Price action parameters", "ANNUALIZED_VOL"))
DRIFT = float(config_object.get("Price action parameters", "DRIFT"))
TIME_HORIZON = float(config_object.get("Price action parameters", "TIME_HORIZON"))
TIME_STEPS_SIZE = float(config_object.get("Price action parameters", "TIME_STEPS_SIZE"))

SEED = int(config_object.get("Simulation parameters", "SEED"))
IS_CONSTANT_PRICE = config_object.getboolean("Simulation parameters", "IS_CONSTANT_PRICE")
PLOT_PRICE_EVOL = config_object.getboolean("Simulation parameters", "PLOT_PRICE_EVOL")
SAVE_PRICE_EVOL = config_object.getboolean("Simulation parameters", "SAVE_PRICE_EVOL")
PLOT_RESERVES = config_object.getboolean("Simulation parameters", "PLOT_RESERVES")
SAVE_RESERVES = config_object.getboolean("Simulation parameters", "SAVE_RESERVES")
PLOT_LIQUID = config_object.getboolean("Simulation parameters", "PLOT_LIQUID")
SAVE_LIQUID = config_object.getboolean("Simulation parameters", "SAVE_LIQUID")


#Initialize pool parameters
np.random.seed(SEED)


#Initialize pool object
Pool = cfmm.ConstantProductCFMM(BUDGET, K)



###### TO DO: Update this with different distributions

# Generate background reference price

#Initialize GBM parameters
T = TIME_HORIZON
dt = TIME_STEPS_SIZE
S0 = INITIAL_REFERENCE_PRICE

t, S = generateGBM(T, DRIFT, ANNUALIZED_VOL, S0, dt)

if IS_CONSTANT_PRICE:
    length = len(S)
    constant_price = []
    for i in range(length):
        constant_price.append(S0)
    S = constant_price



# Prepare storage variables

# Spot price of one asset in terms of the other
X_spot_price_array = []
Y_spot_price_array = []


trade_success_array = []
X_reserves_array = []
Y_reserves_array = []

liquidity_array = []
inefficiency_array = []



##### TO CHECK: SHOULD WE KEEP REF PRICE CONSTANT?

##### TO CHECK: HOW MUCH OF THIS NEEDS TO BE CHANGED WHEN NOT CONSTANT FUNCTION? OR WHEN WE ADD FEES?

for i in range(len(S)):


    # Generate a trade from distribution
    (trade, trade_size) = getSellTrade(TRADE_SIZE_AVERAGE,TRADE_SIZE_VARIANCE,PROBABILITY_OF_TRADE_X)


    # Attempt to make the trade
    if trade == 'SELL X':
        out, price = Pool.swapInAmountX(trade_size, S[i], EPSILON)
        trade_success = True if out!=0 else False
    elif trade == 'SELL Y':
        out, price  = Pool.swapInAmountY(trade_size, S[i], EPSILON)
        trade_success = True if out!=0 else False

    trade_success_array.append(trade_success)


    # Add values to array
    X_spot_price_array.append(1/Pool.getSpotPrice())
    Y_spot_price_array.append(Pool.getSpotPrice())
    X_reserves_array.append(Pool.reserves_x)
    Y_reserves_array.append(Pool.reserves_y)


    ##### TO CHECK - should we make this more an instantaneous derivative? 
    ####### TO CHECK - how does this relate to the other def of liqudity?
    if trade_success and len(Y_spot_price_array) > 1:
        liquidity = (Y_reserves_array[i] - Y_reserves_array[i-1])/(Y_spot_price_array[i] - Y_spot_price_array[i-1])
        liquidity_array.append(liquidity)
        inefficiency = Pool.reserves_y/liquidity
        inefficiency_array.append(inefficiency)


    max_index = i




# Show Data:


######### TO DO - look at trade success evolution time
print("Trade success: {}%".format(100*sum(trade_success_array)/len(S)))


if PLOT_PRICE_EVOL: 
    plt.plot(t[0:max_index], S[0:max_index], label = "Reference price")
    plt.plot(t[0:max_index], X_spot_price_array[0:max_index], label = "Spot Price")
    plt.title("Reference price vs Spot Price of X")
    plt.xlabel("Time steps (years)")
    plt.ylabel("Price (USD)")
    plt.legend(loc='best')
    params_string = 'params_tbd'
    filename = 'price_evol_'+params_string+'.svg'
    plt.plot()
    if SAVE_PRICE_EVOL:
        plt.savefig('sim_results/'+filename)
    plt.show(block = False)



if PLOT_RESERVES:
    plt.figure()
    plt.plot(t[0:max_index], X_reserves_array[0:max_index], label = "X Reserves")
    plt.plot(t[0:max_index], Y_reserves_array[0:max_index], label = "Y Reserves")
    plt.title("Reserves over Time")
    plt.xlabel("Time steps (years)")
    plt.ylabel("Price (USD)")
    plt.legend(loc='best')
    params_string = 'params_tbd'
    filename = 'reserves'+params_string+'.svg'
    plt.plot()
    if SAVE_RESERVES:
        plt.savefig('sim_results/'+filename)
    plt.show(block = True)


if PLOT_LIQUID: 
    plt.figure()
    t_mask = np.where(trade_success_array[1:max_index+1])
    print(len(t_mask[0]))
    print(len(liquidity_array))
    plt.plot(t[t_mask], liquidity_array, label = "Liquidity")
    plt.plot(t[t_mask], inefficiency_array, label = "Inefficiency")
    plt.title("liquidity and inefficiency")
    plt.xlabel("Time steps (years)")
    plt.ylabel("Values")
    plt.legend(loc='best')
    params_string = 'params_tbd'
    filename = 'liquidity'+params_string+'.svg'
    plt.plot()
    if SAVE_LIQUID:
        plt.savefig('sim_results/'+filename)
    plt.show(block = True)


