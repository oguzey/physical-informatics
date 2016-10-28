from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
# do some fancy 3D plotting
from mpl_toolkits.mplot3d import Axes3D


def make_figure(name):
    fig = plt.figure()
    fig.set_size_inches(15, 7)
    fig.canvas.set_window_title(name.capitalize())


def Lorenz(state, t):
    # unpack the state vector
    x = state[0]
    y = state[1]
    z = state[2]

    # these are our constants
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0

    # compute state derivatives
    xd = sigma * (y - x)
    yd = (rho - z) * x - y
    zd = x * y - beta * z

    # return the state derivatives
    return [xd, yd, zd]


state0 = [1.0, 1.0, 1.0]
t = np.arange(0.0, 1250.0, 0.1)
state = odeint(Lorenz, state0, t)
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(state[:, 0], state[:, 1], state[:, 2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')

make_figure('Y1(t)')
plt.title('Y1(t)')
plt.plot(state[:, 0], state[:, 1], 'o-', label='Y1(t)')


plt.show()
