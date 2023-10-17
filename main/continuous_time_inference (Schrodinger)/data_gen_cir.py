"""
We use the analytical solution to the CIR model to generate datapoints to train a PINN to price a zero-coupon bond
parameters that are to be used for the equation of thej short-term rate
b = 0.8
a = 0.1
sigma=0.005
j = 100
dr = b(a-r)dt + sigma*(sqrt(r))dB
r = 0.01 to 0.1 yearly interest rate
time from 0 to 1
"""

# C(s) function s = T-t 
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def C(a, b, sigma, s, b_sq, sig_sq, coeff, eta, eta_sq, const_a):
    c_val = (b / sig_sq) + math.sqrt((2 / sig_sq) + (b_sq / (sig_sq**2)))*((1 + eta*math.exp(coeff*s))/(1 - eta*math.exp(coeff*s)))
    return c_val

def A(a, b, sigma, s, b_sq, sig_sq, coeff, eta, eta_sq, const_a):
    exp_term = math.exp(coeff*s)
    a_val = ((a*b_sq*s) / sig_sq) + ((a*b) / sig_sq)*math.log(abs((eta_sq*exp_term)/((1 - eta*exp_term)**2))) + const_a
    return a_val


def generate_data(a, b, sigma):
    b_sq = b**2
    sig_sq = sigma**2
    coeff = math.sqrt(2*sig_sq + b_sq)
    eta = (b + coeff) / (b - coeff)
    eta_sq = eta**2
    const_a = -((2*a*b) / sig_sq)*math.log(abs(eta / (1 - eta)))
    #some pre-processing for creating the data mesh

    r_range = np.linspace(0.01, 0.1, 100)  # 5 points between 0 and 1
    t_range = np.linspace(0, 1, 100)  # 4 points between 0 and 2

    # Create the mesh grid
    R, T = np.meshgrid(r_range, t_range)
    
    z_range = []
    
    N, M = R.shape
    for i in range(N):
        for j in range(M):
            r = R[i][j]
            t = T[i][j]
            a_val = A(a, b, sigma, t, b_sq, sig_sq, coeff, eta, eta_sq, const_a)
            c_val = C(a, b, sigma, t, b_sq, sig_sq, coeff, eta, eta_sq, const_a)
            z_range.append(math.exp(a_val + r*c_val))
    
    R = R.flatten()
    T = T.flatten()
    
    fig = plt.figure()
    
    ax = fig.add_subplot(projection = '3d')

    
    ax.scatter(R, T, z_range, marker='.')
    
    ax.set_xlabel('Interest rate')
    ax.set_ylabel('Time till Expiry')
    ax.set_zlabel('Bond Price')

    plt.show()
    
    return R, T, z_range

    
    
    
b = 0.8
a = 0.1
sigma=0.005
    
r, t, z = generate_data(a, b, sigma)


data_array = np.stack((r, t, z), axis = 1)
columns = ['rate', 'time', 'price']

df = pd.DataFrame(data_array, columns=columns)

df.to_csv('cir_data.csv', index=False)
print(df)


