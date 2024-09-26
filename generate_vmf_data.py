import numpy as np
from scipy.stats import vonmises_fisher
from tqdm import tqdm

# Generate data
np.random.seed(0)
kappa_values = [100, 500, 1000, 5000, 10000, 20000, 50000, 100000]
dimension_values = [500, 1000, 5000, 10000, 20000, 100000]
n_samples = 1000

print('Generating data...')
for dimension in dimension_values:
    mu = np.zeros(dimension)
    mu[0] = 1
    for kappa in tqdm(kappa_values, desc=f'Dimension: {dimension:6d}'):
        data = vonmises_fisher.rvs(kappa=kappa, mu=mu, size=n_samples)
        np.save(f'vmf_data/kappa_{kappa}_dim_{dimension}.npy', data)
    print('Finsihed generating data for dimension:', dimension)
print('Data generation complete!')
