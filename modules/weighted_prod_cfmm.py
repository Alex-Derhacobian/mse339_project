"""
Contains the necessary AMM logic.
"""

import math
from math import inf
import scipy
from scipy.stats import norm
from scipy import optimize
import numpy as np

from modules.utils import nonnegative, quantilePrime, blackScholesCoveredCallSpotPrice

EPSILON = 1e-8


class ConstantProductCFMM(object):
    """
    A class to represent a two-tokens AMM with the covered call trading function.

    Attributes
    ___________

    reserves_risky: float
        the reserves of the AMM pool in the risky asset
    reserves_riskless: float
        the reserves of the AMM pool in the riskless asset
    tau: float
        the time to maturity for this pool in the desired units
    K: float
        the strike price for this pool
    sigma: float
        the volatility for this pool, scaled to be consistent with the unit of tau (annualized if tau is in years etc)
    invariant: float
        the invariant of the CFMM
    """

    def __init__(self, initial_x, k, alpha):
        """
        Initialize the AMM pool with a starting risky asset reserve as an
        input, calculate the corresponding riskless asset reserve needed to
        satisfy the trading function equation.
        """
        self.reserves_x = initial_x
        self.K = k
        self.alpha = alpha
        self.reserves_y = k / (self.reserves_x ** self.alpha)
        self.fee = 0

    def getXGivenY(self, Y):
        return self.K / (self.reserves_x ** self.alpha)

    def getYGivenX(self, X):
        return (self.K / self.reserves_y) ** (1/ alpha)

    def swapInAmountX(self, amount_in, reference_price, epsilon):
        """
        Swap in some amount of the risky asset and get some amount of the riskless asset in return.

        Returns: 

        amount_out: the amount to be given out to the trader
        effective_price_in_risky: the effective price of the executed trade
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_y = self.getYGivenX(self.reserves_x + gamma * amount_in)
        amount_out = self.reserves_y - new_reserves_y

        exchange_price = self.reserves_y / self.reserves_x 
        slippage = exchange_price / reference_price

        if slippage > (1 + epsilon):
            amount_out = 0
            effective_price_in_x = 0
            return amount_out, effective_price_in_x
        else:
            self.reserves_x += amount_in
            self.reserves_y -= amount_out
            assert nonnegative(new_reserves_y)
            # Update invariant
            effective_price_in_x = amount_out / amount_in
            return amount_out, effective_price_in_x

    def swapInAmountY(self, amount_in, reference_price, epsilon):
        """
        Swap in some amount of the riskless asset and get some amount of the risky asset in return.

        Returns:

        amount_out: the amount to be given to the trader
        effective_price_in_riskless: the effective price the trader actually paid for that trade
        denominated in the riskless asset
        """
        assert nonnegative(amount_in)
        gamma = 1 - self.fee
        new_reserves_x = self.getXGivenY(self.reserves_y + gamma * amount_in)
        amount_out = self.reserves_x - new_reserves_x

        exchange_price = self.reserves_y / self.reserves_x 
        slippage = exchange_price / reference_price

        if slippage > (1 + epsilon):
            amount_out = 0
            effective_price_in_y = 0
            return amount_out, effective_price_in_y
        else:
            self.reserves_y += amount_in
            self.reserves_x -= amount_out
            assert nonnegative(new_reserves_x)
            # Update invariant
            effective_price_in_y = None
            if amount_in == 0:
                effective_price_in_y = inf
            else:
                effective_price_in_y = amount_in / amount_out
            return amount_out, effective_price_in_y

    def getSpotPrice(self):
        """
        Get the current spot price (ie "reported price" using CFMM jargon) of
        the risky asset, denominated in the riskless asset, only exact in the
        no-fee case.
        """
        # TODO ASK
        return self.reserves_y / self.reserves_x

