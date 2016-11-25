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


def get_neuron_states(I_arg, times, state_init, delta_t):
    peak_count = 0
    v = np.empty_like(times)
    u = np.empty_like(times)
    v[0], u[0] = state_init

    for i in range(len(times) - 1):
        if v[i] >= v_thr:
            v[i + 1] = c
            u[i + 1] = u[i] + d
            peak_count += 1
        else:
            dv = f_func(v[i]) - u[i] + I_arg
            du = a * (b * v[i] - u[i])
            v[i + 1] = v[i] + dv * delta_t
            u[i + 1] = u[i] + du * delta_t

    return v, u, peak_count


def calculate_and_draw_states_neuron(I_arg, T_ini=0.0, T_end=1000.0, delta_t=0.01,
                                     v_init=-70., u_init=-20.):
    t = np.arange(T_ini, T_end, delta_t)
    v, u, peak_count = get_neuron_states(I_arg, t, (v_init, u_init), delta_t)
    draw_plot_xy(t, v, 'V(t) for I = {}'.format(I_arg), xlabel='time', ylabel='v')
    draw_plot_xy(t, u, 'U(t) for I = {}'.format(I_arg), xlabel='time', ylabel='u')


def get_frequency_of_peaks(I_arg, T_ini=0.0, T_end=1000.0, delta_t=0.01,
                           v_init=-70., u_init=-20.):
    t = np.arange(T_ini, T_end, delta_t)
    v, u, peak_count = get_neuron_states(I_arg, t, (v_init, u_init), delta_t)
    return peak_count / (T_end - T_ini)


I_critical = ((k2 - b)**2) / (4 * k1) - k3
logger.info('I critical is equal to %0.3f', I_critical)

# step 3
calculate_and_draw_states_neuron(0)
calculate_and_draw_states_neuron(0.5)
calculate_and_draw_states_neuron(I_critical)
calculate_and_draw_states_neuron(2)

# step 4

#step 5
Is = [3, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95]
frequencies = [get_frequency_of_peaks(i) for i in Is]
draw_plot_xy(Is, frequencies, 'Frequency of peaks', xlabel='I', ylabel='frequency')

show_all_plots()
