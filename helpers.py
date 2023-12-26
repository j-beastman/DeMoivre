from mpmath import mp, binomial
from scipy.stats import norm
import numpy as np

# Give better name for function
def find_p_x_equals_k(k: int, n: int, p: float):
    '''
    This function takes in 3 inputs and produces a numerical output
        between 1 and 0 representing the probability that the random
        variable X = k, given a population size of n, and probability of
        success p
    Input:
        k: int
        n: int -- Population size, the number of independent Bernoulli
            trials
        p: float -- probability of success of one trial, independent
            of other trials
    '''
    q = 1 - p
    n_choose_k = binomial(n, k)
    return n_choose_k * (p ** k) * (q ** (n - k))

def probability_binomial(a: int, b: int, n: int, p: float):
    '''
    This function computes the probability that a random variable 
        modeled by the binomial distribution, falls between the values
        of a and b, where a is the lower bound, b is the upper bound.
    Input:
        a: int = Lower bound of the range, must be >= 0
        b: int = Upper bound of the range, must be <= n
        n: int = Population size, the number of independent Bernoulli
            trials
        p: float -- probability of success of one trial, independent
            of other trials
    '''
    sum = mp.mpf(0.0)
    q = 1 - p
    for value in range(a, b + 1, 1):
        n_choose_k = binomial(n, value)
        term = n_choose_k * (p ** value) * (q ** (n - value))
        sum += term
    return sum

def approx_P_binomial(a: int, b: int, n: int, p: float):
    '''
    This function computes the probability that a random variable 
        modeled by the binomial distribution, falls between the values
        of a and b by approximating the random variable with the normal
        distribution.
    Input:
        a: int = Lower bound of the range, must be >= 0
        b: int = Upper bound of the range, must be <= n
        n: int = Population size, the number of independent Bernoulli
            trials
        p: float -- probability of success of one trial, independent
            of other trials
    '''
    # First, convert n and p into μ and σ
    mu = n * p
    sigma = np.sqrt(mu * (1 - p))

    upper_bound_z_score = (b - mu + 0.5) / sigma
    lower_bound_z_score = (a - mu - 0.5) / sigma

    phi_upper = norm.cdf(upper_bound_z_score)
    phi_lower = norm.cdf(lower_bound_z_score)

    return phi_upper - phi_lower