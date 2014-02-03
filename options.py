from numpy import *
from scipy.stats import binom

class Option(object):
    """Enumerables for option type"""
    CALL = 1
    PUT = 2
        

def call_price(R, S0, K, u):
    """compute call option price 1-period binominal model"""
    d = 1 / u
    rates = array([u, d])
    payoffs = maximum(rates * S0 - K, zeros(2))
    q = (R - d) / (u - d)
    Q = array([q, 1 - q])
    return dot(payoffs, Q) / R
    
def convert_bs_2_binomial(T, sigma, r, c, n):
    """convert Black-Scholes parameters to parameters of n-period binomial model"""
    u = exp(sigma * sqrt(T / n))
    d = 1 / u
    q = (exp((r - c) * T / n) - d) / (u - d)
    R = exp(r * T / n)
    return [u, q, R]

def generate_binominal_matrix(u, d, n):
    """generate binominal matrix for stock price. u, d - parameters, n - number of periods"""
    return vstack([ eye(N = n, M = n + 1, k = 1) * u, eye(N = 1, M = n + 1, k = n) * d ])    

def spot_prices_binomial(s0, u, n):
    """generate spot prices matrix for n-period binomial model with parameter u"""
    m = n + 1
    P = zeros(shape=(m, m))
    M = generate_binominal_matrix(u = u, d = 1 / u, n = n) 
    P[m - 1, 0] = float(s0)
    for i in range(1, m):
        P[:, i] = dot(M, P[:, i - 1])
    return P
    
def option_payoff_binomial(S, type, K):
    """payoff of option"""
    P = (S - K) if type == Option.CALL else (K - S)
    return maximum(P, 0.0)
    
def option_price(P, R, q):
    pv = P[:, -1]
    n = size(pv) - 1
    b = binom(n, 1 - q)
    vf = vectorize(lambda k: b.pmf(k))
    qv = vf(range(n + 1))
    return dot(qv, pv) / R ** n
    
def compute_price(s0, T, sigma, r, c, n, type, K):
    u, q, R = convert_bs_2_binomial(T, sigma, r, c, n)
    S  = spot_prices_binomial(s0, u, n)
    P = option_payoff_binomial(S, type, K)
    return option_price(P, R, q)
    
    
    