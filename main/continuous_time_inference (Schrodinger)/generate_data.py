import numpy as np
import matplotlib.pyplot as plt

def tridiagonalGaussianElimination(A, d):
    n = len(d)
    
    #forward elimination
    for i in range(1, n):
        alpha = A[i][i-1] / A[i-1][i-1]
        # A[i][i-1] = 0
        A[i][i] = A[i][i] - alpha*A[i-1][i]
        d[i] = d[i] - alpha*d[i-1]
        
    x = np.zeros((n, 1))
    x[n-1] = d[n-1] / A[n-1][n-1]
    
    for i in range(n-2, -1, -1):
        x[i] = ((d[i] - A[i][i+1]*x[i+1]) / A[i][i])
    
    return x

def calculate_d_val(t, mesh, mesh_params, k, h, I, N, sigma, a, b):
    d = np.zeros((I+1, 1))
    
    xmin = mesh_params["x_min"]
    xmax = mesh_params["x_max"]
    d[0] = -(k*(sigma**2)*xmin + k*h*a*(b-xmin))*mesh[t][1] + (4*(h**2) - -2*(sigma**2)*xmin - -2*(h**2)*k*xmin)*mesh[t][0] + -(k*(sigma**2)*xmax - k*h*a*(b-xmax))
    for i in range(1, I):
        x = xmin + i*h
        d[i] = -(k*(sigma**2)*x + k*h*a*(b-x))*mesh[t][i+1] + (4*(h**2) - -2*(sigma**2)*x - -2*(h**2)*k*x)*mesh[t][i] + -(k*(sigma**2)*x - k*h*a*(b-x))*mesh[t][i-1]
    d[I] = -(k*(sigma**2)*xmax + k*h*a*(b-xmax))*0 + (4*(h**2) - -2*(sigma**2)*xmax - -2*(h**2)*k*xmax)*mesh[t][i] + -(k*(sigma**2)*xmax - k*h*a*(b-xmax))*mesh[t][i-1]
    
    return d

def calculate_A_matrix(t, mesh, mesh_params, k, h, I, N, sigma, a, b):
    A = np.zeros((I+1, I+1))
    xmin = mesh_params["x_min"]
    xmax = mesh_params["x_max"]
    A[0][0] = (4*(h**2) + -2*(sigma**2)*xmin + -2*(h**2)*k*xmin)
    A[0][1] = (k*(sigma**2)*xmin - k*h*a*(b-xmin))
    for i in range(1, I):
        x = xmin + i*h
        aa = (k*(sigma**2)*x + k*h*a*(b-x))
        cc = (k*(sigma**2)*x - k*h*a*(b-x))
        bb = (4*(h**2) + -2*(sigma**2)*x + -2*(h**2)*k*x)
        
        A[i][i-1] = aa
        A[i][i] = bb
        A[i][i+1] = cc
    
    A[I][I] = (4*(h**2) + -2*(sigma**2)*xmax + -2*(h**2)*k*xmax)
    A[I][I-1] = (k*(sigma**2)*xmax + k*h*a*(b-xmax))
    
    return A
    

I = 100
N = 100

mesh_params = {
    "t_min" : 0,
    "t_max" : 1,
    "x_min" : 0.01,
    "x_max" : 0.07,
}
k = (mesh_params["t_max"] - mesh_params["t_min"]) / N
h = (mesh_params["x_max"] - mesh_params["x_min"]) / I

sigma = 0.02
a = 0.5
b = 0.05


mesh_values = []
mesh_values.append(np.ones((I+1, 1)))

for i in range(1, N+1):
    A = calculate_A_matrix(0, mesh_values, mesh_params, k, h, I, N, sigma, a, b)
    d = calculate_d_val(0, mesh_values, mesh_params, k, h, I, N, sigma, a, b)
    
    x = tridiagonalGaussianElimination(A, d)
    # mesh_values.append(x)
    mesh_values.insert(0, x)


# print(mesh_values)

fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(projection='3d')
xarr, tarr, zarr = [], [], []
for i in range(0, I+1):
    for j in range(0, N+1):
        x = mesh_params["x_min"] + i*h
        t = mesh_params["t_min"] + j*k
        z = mesh_values[j][i]
        xarr.append(x)
        tarr.append(t)
        zarr.append(z)
ax.scatter(tarr, xarr, zarr, marker='.')

ax.set_xlabel("time")
ax.set_ylabel("rate")
ax.set_zlabel("F")

plt.show()