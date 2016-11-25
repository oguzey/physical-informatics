from scipy.integrate import odeint
import numpy as np
from draw import draw_plot_xy, show_all_plots
from custom_logger import logger


a = 0.02
b = 0.25
c = -65.0
d = 6.0

k1 = 0.04
k2 = 5
k3 = 140

v_thr = 30


def f_func(v_arg):
    return k1 * v_arg ** 2 + k2 * v_arg + k3


def get_neuron_states(I_arg, times, state_init):

    def behavior_neuron(state, t):
        v, u = state
        dv = f_func(v) - u + I_arg
        du = a * (b * v - u)
        if dv >= v_thr:
            return [c, du + d]
        return [dv, du]

    return odeint(behavior_neuron, state_init, times)


def calculate_and_draw_states_neuron(I_arg, T_ini=0.0, T_end=1000.0, delta_t=0.01,
                                     v_init=-70., u_init=-20.):
    t = np.arange(T_ini, T_end, delta_t)
    states = get_neuron_states(I_arg, t, [v_init, u_init])
    draw_plot_xy(t, states[:, 0], 'V(t) for I = {}'.format(I_arg))
    draw_plot_xy(t, states[:, 1], 'U(t) for I = {}'.format(I_arg))


I_critical = ((k2 - b)**2) / (4 * k1) - k3
logger.info('I critical is equal to %0.3f', I_critical)

calculate_and_draw_states_neuron(0)
calculate_and_draw_states_neuron(50)
calculate_and_draw_states_neuron(100)
calculate_and_draw_states_neuron(I_critical)
calculate_and_draw_states_neuron(I_critical + 10)
calculate_and_draw_states_neuron(I_critical - 10)
show_all_plots()
