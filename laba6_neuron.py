from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

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


def get_neuron_states(I_arg, times):

    def behavior_neuron(state, t):
        v, u = state
        dv = f_func(v) - u + I_arg
        du = a * (b * v - u)
        return [dv, du]

    state_init = [-70., -20.]  # v(0), u(0)
    return odeint(behavior_neuron, state_init, times)


delta_t = 0.01
T = 250.0
T_ini = 0.0
t = np.arange(T_ini, T, delta_t)
states = get_neuron_states(1, t)
