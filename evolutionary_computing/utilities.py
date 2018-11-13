import numpy as np

def rand_real(n=1):
	m = sys.maxint
	np.random.uniform(-m, m, n)

def rand_normal(loc=0.0, scale=1.0, size=None):
    return np.random.normal(loc, scale, size)

def rand_power_law():
    alpha = 2.
    norm_rand = rand_normal()
    dir = np.random.choice([-1,1])
    
    return dir * (1 - norm_rand) ** (1/(1-alpha))