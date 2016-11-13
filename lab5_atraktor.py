from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np
# do some fancy 3D plotting
from mpl_toolkits.mplot3d import Axes3D
import logging
import sys
from sympy import sqrt
from array import array

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


def extend_state_and_time(system_states, times, delta_time, new_time):
    if new_time <= times[-1]:
        logger.warn('Value of new time is less that already exists.')
        return system_states, times

    new_times = np.arange(times[-1], new_time, delta_time)
    new_states = odeint(Lorenz, system_states[-1], new_times)
    res_states = np.append(system_states, new_states[1:], axis=0)
    res_times = np.append(times, new_times[1:], axis=0)
    return res_states, res_times


def get_state_by_time(timestamp):
    """
    Function return state using timestamp as parameter. If the timestamp is greater than max time the function will be
    generate new value of state.
    USES GLOBAL VARIABLES - states, t, delta_t
    :param timestamp: specific time stamp
    :return: state as array [Y1, Y2, Y3]
    """
    global states
    global t
    global delta_t

    if timestamp > t[-1]:
        logger.debug('Start generate new states for time %f plus 100', timestamp)
        states, t = extend_state_and_time(states, t, delta_t, timestamp + 100)

    position = int(timestamp / delta_t)
    assert len(states[position]) == 3
    return states[position]


def get_state_by_pos(position):
    """
    Function return state using position as parameter. If the position is greater than max len the function will be
    generate new value of state.
    USES GLOBAL VARIABLES - states, t, delta_t
    :param position: specific position
    :return: state as array [Y1, Y2, Y3]
    """
    global states
    global t
    global delta_t

    try:
        return states[position]
    except IndexError:
        new_timestamp = position * delta_t
        logger.debug('Start generate new states for time %f plus 100', new_timestamp)
        states, t = extend_state_and_time(states, t, delta_t, new_timestamp + 100)
        return states[position]


def calculate_R(tau):
    global delta_t
    global T_ini
    global T

    start = int(T_ini / delta_t)
    N = int(T / delta_t)
    tau_pos = int(tau / delta_t)

    R = 0
    for k in range(start, N + 1, 1):
        # R += get_state_by_time(k * delta_t)[0] * get_state_by_time(k * delta_t + tau)[0]
        R += get_state_by_pos(k)[0] * get_state_by_pos(k + tau_pos)[0]

    R /= N
    return R


def calculate_C(r, points, diff_func):
    global N

    start = 0  # int(T_ini / delta_t)
    res = 0
    for k in range(start, N + 1, 1):
        for j in range(start, N + 1, 1):
            res += bool((r - diff_func(points, j, k)) > 0)
    res /= N**2

    return res


def diff_onedimensional_points(points, j, k):
    assert isinstance(points, array)

    return abs(points[j] - points[k])


def diff_twodimensional_points(points, j, k):
    assert isinstance(points, tuple) and len(points) == 2

    return sqrt((points[0][j] - points[0][k]) ** 2 + (points[1][j] - points[1][k]) ** 2)


def diff_threedimensional_points(points, j, k):
    assert isinstance(points, tuple) and len(points) == 3

    return sqrt((points[0][j] - points[0][k]) ** 2 + (points[1][j] - points[1][k]) ** 2 + (points[2][j] - points[2][k]) ** 2)


def draw_original_system(system_states):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(system_states[:, 0], system_states[:, 1], system_states[:, 2])
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')


def draw_plot_xy(x, y, name):
    make_figure(name)
    plt.title(name)
    plt.plot(x, y, 'o-', label=name)
    plt.grid(b=True, which='major', color='grey', linestyle='--')

delta_t = 0.002
T = 250.0
T_ini = 0.0
t = np.arange(T_ini, T, delta_t)
states = odeint(Lorenz, [1.0, 1.0, 1.0], t)

# draw_original_system(states)

# step 2
draw_plot_xy(t, states[:, 0], 'Y1(t)')
# redefine Tini as 18
T_ini = 18.0

# cut states from [0, T_ini]
start = int(T_ini / delta_t)
logger.debug('New start %i', start)
cropped_states = states[start:]
cropped_t = t[start:]
draw_plot_xy(cropped_t, cropped_states[:, 0], 'Y1(t) cropped')


# step 3
N_max = int(T / delta_t)
logger.info('N_max equal to %.0f', N_max)

# step 4
taus = range(1, 100, 1)
Rs = [calculate_R(x) for x in taus]
draw_plot_xy(taus, Rs, 'R(T)')

tau_optimal = Rs.index(min(Rs))
logger.info('Tau optimal equal to %d', tau_optimal)

# step 5
tmp = int(N_max - 2 * tau_optimal / delta_t)
assert tmp > 10000
N = 100
logger.info('N should be less than %d', tmp)
logger.info('N equal to %d', N)

tau_optimal_pos = int(tau_optimal / delta_t)
y1_s = array('d', [get_state_by_pos(x)[0] for x in range(N + 1)])
y1_s_tau = array('d', [get_state_by_pos(x + tau_optimal_pos)[0] for x in range(N + 1)])
y1_s_double_tau = array('d', [get_state_by_pos(x + 2 * tau_optimal_pos)[0] for x in range(N + 1)])

# step 6
logger.debug('Calculating C(r)')
rs = [x for x in range(11)]   # [0, 10]
#Cs_one_dim = [calculate_C(x, y1_s, diff_onedimensional_points) for x in rs]
logger.debug('Done C(r) one')
Cs_two_dim = [calculate_C(x, (y1_s, y1_s_tau), diff_twodimensional_points) for x in rs]
logger.debug('Done C(r) two')
Cs_three_dim = [calculate_C(x, (y1_s, y1_s_tau, y1_s_double_tau), diff_threedimensional_points) for x in rs]
logger.debug('Done C(r) three')

#draw_plot_xy(rs, Cs_one_dim, 'C(r) one')
draw_plot_xy(rs, Cs_two_dim, 'C(r) two')
draw_plot_xy(rs, Cs_three_dim, 'C(r) three')

plt.show()
