from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
# do some fancy 3D plotting
from mpl_toolkits.mplot3d import Axes3D
import logging
import sys

sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(sh)


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


def draw_original_system(system_states):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(system_states[:, 0], system_states[:, 1], system_states[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def draw_y1_t(states_y1):
    make_figure('Y1(t)')
    plt.title('Y1(t)')
    plt.plot(t, states_y1, 'o-', label='Y1(t)')
    plt.grid(b=True, which='major', color='grey', linestyle='--')


delta_t = 0.02
T = 250.0
T_ini = 0.0
t = np.arange(T_ini, T, delta_t)
states = odeint(Lorenz, [1.0, 1.0, 1.0], t)

# draw_original_system(states)

# step 2
draw_y1_t(states[:, 0])
# redefine Tini as 18
T_ini = 18.0

# step 3
N_max = T / delta_t
logger.info('N max equal to %f', N_max)

# step 4



plt.show()
