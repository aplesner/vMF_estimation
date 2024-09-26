import os
import time
import re
from typing import Union

import numpy as np
from scipy.optimize import minimize
from scipy.special import ive

time_requirement = 0.5

def time_method(method, data: np.ndarray) -> float:
    rounds = 0
    start = time.time()
    while True:
        kappa_hat = method(data)
        rounds += 1
        if time.time() - start > time_requirement:
            break
    average_time = (time.time() - start)/rounds
    print(f'Average time: {1000*average_time:.2f} ms over {rounds} rounds')
    return kappa_hat


def method_error(kappa_hat: float, kappa: float) -> None:
    error = np.abs(kappa_hat - kappa)
    relative_error = error / kappa
    print(f'Estimated kappa: {kappa_hat:.2f}, error: {error:.2e}, relative error: {relative_error:.2e}')


def Ap_scipy(kappa: float, p: int) -> float:
    return ive(p/2, kappa) / ive(p/2 - 1, kappa)


def log_iv_scipy(kappa: float, v: float) -> float:
    return np.log(ive(v, kappa)) + kappa


def log_iv_jac_scipy(kappa: float, v: float) -> float:
    return v/kappa + ive(v+1, kappa) / ive(v, kappa)

def banerjee_estimation(data: np.ndarray) -> float:
    n, p = data.shape
    R_bar = np.linalg.norm(np.sum(data, axis=0)) / n
    kappa_hat = R_bar * (p - R_bar**2) / (1 - R_bar**2)
    return kappa_hat


def tanabe_estimation(data: np.ndarray, Ap = Ap_scipy) -> float:
    def Phi_2p(kappa: float, R_bar: float, p: int) -> float:
        return R_bar * kappa / Ap(kappa, p)
    
    n, p = data.shape
    R_bar = np.linalg.norm(np.sum(data, axis=0)) / n
    kappa_l = R_bar * (p - 2)/(1-R_bar**2)
    kappa_u = R_bar * p/(1-R_bar**2)
    Phi_2p_kappa_l = Phi_2p(kappa=kappa_l, R_bar=R_bar, p=p)
    Phi_2p_kappa_u = Phi_2p(kappa=kappa_u, R_bar=R_bar, p=p)
    kappa_hat = (kappa_l * Phi_2p_kappa_u - kappa_u * Phi_2p_kappa_l) / ((Phi_2p_kappa_u - Phi_2p_kappa_l) - (kappa_u - kappa_l))
    return kappa_hat


def sra_estimation(data: np.ndarray, Ap = Ap_scipy) -> float:
    n, p = data.shape
    R_bar = np.linalg.norm(np.sum(data, axis=0)) / n
    kappa_prev = R_bar * (p - R_bar**2) / (1 - R_bar**2)
    for _ in range(2):
        Ap_kappa_prev = Ap(kappa_prev, p)
        kappa_next = kappa_prev - (Ap_kappa_prev - R_bar) / (1 - Ap_kappa_prev**2 - (p-1)/kappa_prev * Ap_kappa_prev)
        kappa_prev = kappa_next
    return kappa_next


def ml_estimation(data: np.ndarray, log_iv = log_iv_scipy) -> float:
    x_bar = np.mean(data, axis=0)
    R_bar = np.linalg.norm(x_bar)
    mu = x_bar / R_bar
    muT_data = data @ mu

    def log_likelihood(kappa: float, muT_data: np.ndarray, p: int) -> float:
        log_Cp_kappa = (p/2-1) * np.log(kappa) - (p/2)*np.log(2*np.pi) - log_iv(kappa, p/2-1)
        return np.mean(muT_data * kappa) + log_Cp_kappa
    
    p = data.shape[1]
    kappa_init = 100 # R_bar * (p - R_bar**2) / (1 - R_bar**2)
    res = minimize(lambda kappa: -log_likelihood(kappa, muT_data=muT_data, p=p), x0=kappa_init, bounds=[(0, None)])
    # print(res)
    return res.x[0]


def ml_jac_estimation(data: np.ndarray, log_iv = log_iv_scipy, log_iv_jac = log_iv_jac_scipy) -> float:
    x_bar = np.mean(data, axis=0)
    R_bar = np.linalg.norm(x_bar)
    mu = x_bar / R_bar
    muT_data = data @ mu

    def log_likelihood(kappa: float, muT_data: np.ndarray, p: int) -> float:
        log_Cp_kappa = (p/2-1) * np.log(kappa) - (p/2)*np.log(2*np.pi) - log_iv(kappa, p/2-1)
        return np.mean(muT_data)*kappa + log_Cp_kappa

    def jac(kappa: float, muT_data: np.ndarray, p: int) -> float:
        log_Cp_kappa_jac = (p/2-1) / kappa - log_iv_jac(kappa, p/2-1)
        return np.mean(muT_data) + log_Cp_kappa_jac
    
    p = data.shape[1]
    kappa_init = R_bar * (p - R_bar**2) / (1 - R_bar**2)
    res = minimize(
        lambda kappa: -log_likelihood(kappa, muT_data=muT_data, p=p),
        x0=kappa_init,
        bounds=[(0, None)],
        jac=lambda kappa: -jac(kappa, muT_data=muT_data, p=p)
    )
    return res.x[0]


# Load data paths
data_dir = 'vmf_data'
data_files = [os.path.join(data_dir, data_file) for data_file in os.listdir('vmf_data')]

dimension_pattern = re.compile(r'dim_(\d+)')
kappa_pattern = re.compile(r'kappa_(\d+)')

for i, data_file in enumerate(data_files):
    data = np.load(data_file)
    dimension = int(dimension_pattern.search(data_file).group(1))
    kappa = float(kappa_pattern.search(data_file).group(1))
    
    if dimension != 500:
        continue
    
    print(f'Loaded data for dimension {dimension} and kappa {int(kappa)}')

    # Fit data

    # Banerjee estimation
    print('Banerjee estimation')
    kappa_hat = time_method(banerjee_estimation, data)
    method_error(kappa_hat, kappa)
    print()

    # Tanabe estimation
    print('Tanabe estimation')
    kappa_hat = time_method(tanabe_estimation, data)
    method_error(kappa_hat, kappa)
    print()

    # Sra estimation
    print('Sra estimation')
    kappa_hat = time_method(sra_estimation, data)
    method_error(kappa_hat, kappa)
    print()

    # Maximum likelihood estimation
    print('Maximum likelihood estimation')
    kappa_hat = time_method(ml_estimation, data)
    method_error(kappa_hat, kappa)
    print()

    # Maximum likelihood estimation with Jacobian
    print('Maximum likelihood estimation with Jacobian')
    kappa_hat = time_method(ml_jac_estimation, data)
    method_error(kappa_hat, kappa)
    print()


    print()
    break

    