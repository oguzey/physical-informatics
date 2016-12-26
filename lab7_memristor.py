import numpy as np
import matplotlib.pyplot as plt
import pylab

from memristor import Memristor

SM2_TO_M2 = ((10**(-2))**2)
NM_TO_M = 10**(-9)

D = 10 * NM_TO_M
mu = (10**(-10)) * SM2_TO_M2
nu = 100
w0 = D / 10
R_ON = 1
R_OFF = R_ON * 160
U0 = 1


def U_sin(t, U0, nu):
    return U0 * np.sin(nu * t)


def Phi_sin(t, U0, nu):
    return U0 * (1 - np.cos(nu * t)) / nu


# 1 step
# a
m1_a = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=160, U=lambda t: U_sin(t, U0, nu), Phi=lambda t: Phi_sin(t, U0, nu))

m1_a.task1(plt, N=1000, periods=5)
m1_a.task2(plt, N=1000, periods=5)
m1_a.task3(plt, N=1000, periods=5)

# b
m1_b = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=380, U=lambda t: U_sin(t, U0, nu), Phi=lambda t: Phi_sin(t, U0, nu))
m1_b.task1(plt, N=1000, periods=5)
m1_b.task2(plt, N=1000, periods=5)
m1_b.task3(plt, N=1000, periods=5)

# step 2


def U_sin_sqr(t, U0, nu):
    return U0 * (np.sin(nu * t) ** 2)


def Phi_sin_sqr(t, U0, nu):
    return U0 * (2 * nu * t - np.sin(2 * nu * t)) / (4 * nu)


m2_a = Memristor(D, mu, nu, w0, R_ON=1, R_OFF=160, U=lambda t: U_sin_sqr(t, U0, nu), lambda t: Phi_sin_sqr(t, U0, nu))
m2_a.task1(plt, N=1000, periods=5)
m2_a.task2(plt, N=1000, periods=5)
m2_a.task3(plt, N=1000, periods=5)

m2_b = Memristor(D, mu, nu, w0, R_ON=1, R_OFF=380, U=lambda t: U_sin_sqr(t, U0, nu), lambda t: Phi_sin_sqr(t, U0, nu))
m2_b.task1(plt, N=1000, periods=5)
m2_b.task2(plt, N=1000, periods=5)
m2_b.task3(plt, N=1000, periods=5)
# step 3

m3_a_1 = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=125, U=lambda t: U_sin(t, 1, nu), Phi=lambda t: Phi_sin(t, 1, nu))
m3_a_2 = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=125, U=lambda t: U_sin(t, 2, nu), Phi=lambda t: Phi_sin(t, 2, nu))
m3_a_3 = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=125, U=lambda t: U_sin(t, 4, nu), Phi=lambda t: Phi_sin(t, 4, nu))

m3_a_1.task1(plt, N=1000, periods=5, need_U_t=False)
m3_a_2.task1(plt, N=1000, periods=5, need_U_t=False)
m3_a_3.task1(plt, N=1000, periods=5, need_U_t=False)

m3_b_1 = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, 1, nu), Phi=lambda t: Phi_sin(t, 1, nu))
m3_b_2 = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, 2, nu), Phi=lambda t: Phi_sin(t, 2, nu))
m3_b_3 = Memristor(D=D, mu=mu, nu=nu, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, 4, nu), Phi=lambda t: Phi_sin(t, 4, nu))

m3_b_1.task1(plt, N=1000, periods=5, need_U_t=False)
m3_b_2.task1(plt, N=1000, periods=5, need_U_t=False)
m3_b_3.task1(plt, N=1000, periods=5, need_U_t=False)


# step 4
nu1 = 1
nu2 = 10
nu3 = 100
nu4 = 1000

m4_a_1 = Memristor(D=D, mu=mu, nu=nu1, w0=w0, R_ON=160, R_OFF=1, U=lambda t: U_sin(t, U0, nu1), Phi=lambda t: Phi_sin(t, U0, nu1))
m4_a_2 = Memristor(D=D, mu=mu, nu=nu2, w0=w0, R_ON=160, R_OFF=1, U=lambda t: U_sin(t, U0, nu2), Phi=lambda t: Phi_sin(t, U0, nu2))
m4_a_3 = Memristor(D=D, mu=mu, nu=nu3, w0=w0, R_ON=160, R_OFF=1, U=lambda t: U_sin(t, U0, nu3), Phi=lambda t: Phi_sin(t, U0, nu3))
m4_a_4 = Memristor(D=D, mu=mu, nu=nu4, w0=w0, R_ON=160, R_OFF=1, U=lambda t: U_sin(t, U0, nu4), Phi=lambda t: Phi_sin(t, U0, nu4))

m4_a_1.task1(plt, N=1000, periods=5, need_U_t=False)
m4_a_2.task1(plt, N=1000, periods=5, need_U_t=False)
m4_a_3.task1(plt, N=1000, periods=5, need_U_t=False)

plt.show()

m4_b_1 = Memristor(D=D, mu=mu, nu=nu1, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, U0, nu1), Phi=lambda t: Phi_sin(t, U0, nu1))
m4_b_2 = Memristor(D=D, mu=mu, nu=nu2, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, U0, nu2), Phi=lambda t: Phi_sin(t, U0, nu2))
m4_b_3 = Memristor(D=D, mu=mu, nu=nu3, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, U0, nu3), Phi=lambda t: Phi_sin(t, U0, nu3))
m4_b_4 = Memristor(D=D, mu=mu, nu=nu4, w0=w0, R_ON=1, R_OFF=50, U=lambda t: U_sin(t, U0, nu4), Phi=lambda t: Phi_sin(t, U0, nu4))

m4_b_1.task1(plt, N=1000, periods=5, need_U_t=False)
m4_b_2.task1(plt, N=1000, periods=5, need_U_t=False)
m4_b_3.task1(plt, N=1000, periods=5, need_U_t=False)

plt.show()

plt.show()
