import numpy as np

from kim import Data

np.random.seed(1)

Ns = 100
method='gsa'
metric='it-bins'
sst=True
ntest=100 
alpha=0.05 
bins=10
k=5

def get_samples_1():
    xdata = np.arange(Ns)
    ydata = np.arange(Ns)
    return xdata, ydata

def get_samples_2():
    xdata = np.arange(Ns)
    ydata = np.arange(Ns)
    ydata[[10,20,30,40,50]] = 80
    return xdata, ydata

def get_samples_3():
    data = np.random.uniform(size=(Ns,2))
    x, y = data[:,0], data[:,1]
    return x, y

def get_samples_4():
    x = np.linspace(0, 1, Ns)
    y = x**2 + 2*x + 3
    return x, y

def test_Data():
    x1, y1 = get_samples_1()
    x2, y2 = get_samples_2()
    x3, y3 = get_samples_3()
    x4, y4 = get_samples_4()
    xdata = np.array([x1, x2, x3, x4]).T
    ydata = np.array([y1, y2, y3, y4]).T

    data = Data(xdata, ydata)
    
    assert data.Ns == Ns
    assert data.Nx == 4
    assert data.Ny == 4

    data.calculate_sensitivity(
        method, metric, sst, ntest, alpha, k=k
    )

    assert data.sensitivity_config['method'] == method
    assert not data.sensitivity_mask[2,2]
    assert data.sensitivity_mask[-1,-1]
    assert data.sensitivity_mask[0,0]

    # print(data.sensitivity)
    # print(data.sensitivity_mask)