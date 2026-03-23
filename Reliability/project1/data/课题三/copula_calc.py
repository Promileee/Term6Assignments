import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize

df_final = pd.read_csv('cleaned_data.csv')
t_data = df_final['fail_time'].dropna().sort_values().values
x_data = df_final['mileage_val'].dropna().sort_values().values

u_emp = stats.rankdata(t_data)/(len(t_data)+1)
v_emp = stats.rankdata(x_data)/(len(x_data)+1)

p_gamma = stats.gamma.fit(t_data, floc=0)
u_par = stats.gamma.cdf(t_data, p_gamma[0], scale=p_gamma[2])

p_wei = stats.weibull_min.fit(x_data, floc=0)
v_par = stats.weibull_min.cdf(x_data, p_wei[0], scale=p_wei[2])

u_par = np.clip(u_par, 1e-5, 1-1e-5)
v_par = np.clip(v_par, 1e-5, 1-1e-5)

def nll_clayton(theta, u, v):
    if theta <= 0: return np.inf
    t1 = (u**(-theta) + v**(-theta) - 1.0)
    if np.any(t1 <= 0): return np.inf
    pdf = (1.0 + theta) * (u*v)**(-theta - 1.0) * (t1)**(-2.0 - 1.0/theta)
    return -np.sum(np.log(np.clip(pdf, 1e-20, None)))

def nll_gumbel(theta, u, v):
    if theta < 1: return np.inf
    lu = -np.log(u)
    lv = -np.log(v)
    t1 = lu**theta + lv**theta
    C = np.exp(-t1**(1.0/theta))
    pdf = C * (u*v)**(-1) * (lu*lv)**(theta-1) * t1**(2/theta - 2) * (t1**(1/theta) + theta - 1)
    return -np.sum(np.log(np.clip(pdf, 1e-20, None)))

def nll_frank(theta, u, v):
    if theta == 0: return np.inf
    num = -theta * (np.exp(-theta) - 1) * np.exp(-theta*(u+v))
    den = (np.exp(-theta*u) - 1) * (np.exp(-theta*v) - 1) + (np.exp(-theta) - 1)
    pdf = num / (den**2)
    return -np.sum(np.log(np.clip(pdf, 1e-20, None)))

def nll_gaussian(rho, u, v):
    if not -0.99 < rho < 0.99: return np.inf
    x = stats.norm.ppf(u)
    y = stats.norm.ppf(v)
    term = (x**2 + y**2 - 2*rho*x*y) / (2*(1-rho**2))
    pdf = (1.0 / np.sqrt(1-rho**2)) * np.exp(-term + (x**2+y**2)/2)
    return -np.sum(np.log(np.clip(pdf, 1e-20, None)))

def fit_copulas(u, v):
    res_c = minimize(lambda p: nll_clayton(p[0], u, v), [2.0], bounds=[(1e-3, 50)])
    res_g = minimize(lambda p: nll_gumbel(p[0], u, v), [2.0], bounds=[(1.001, 50)])
    res_f = minimize(lambda p: nll_frank(p[0], u, v), [2.0], bounds=[(-50, 50)])
    res_gauss = minimize(lambda p: nll_gaussian(p[0], u, v), [0.5], bounds=[(-0.99, 0.99)])
    print(f"Clayton: {res_c.x[0]:.4f}, NLL: {res_c.fun:.2f}, AIC: {2*1 + 2*res_c.fun:.2f}")
    print(f"Gumbel: {res_g.x[0]:.4f}, NLL: {res_g.fun:.2f}, AIC: {2*1 + 2*res_g.fun:.2f}")
    print(f"Frank: {res_f.x[0]:.4f}, NLL: {res_f.fun:.2f}, AIC: {2*1 + 2*res_f.fun:.2f}")
    print(f"Gaussian: {res_gauss.x[0]:.4f}, NLL: {res_gauss.fun:.2f}, AIC: {2*1 + 2*res_gauss.fun:.2f}")

print("--- Empirical ---")
fit_copulas(u_emp, v_emp)
print("--- Parametric ---")
fit_copulas(u_par, v_par)
