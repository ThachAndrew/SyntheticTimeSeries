import numpy as np

def add_gaussian_noise(series, mu, sigma, seed):
    np.random.seed(seed=seed)
    return series + np.random.normal(mu, sigma * np.std(series), size=len(series))
