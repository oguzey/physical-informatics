from collections import namedtuple
import numpy as np

def get_roots(a, b, c, d):

    p = (3*a*c - b*b) / (3*a*a)
    q = (2*b**3 - 9*a*b*c + 27*a*a*d) / (27*a**3)

    Q = (p / 3)**3 + (q / 2)**2

    print("\tQ = {}".format(Q))

    alfa = (-q/2 + Q**0.5)**(1 / 3)
    beta = (-q/2 - Q**0.5)**(1 / 3)

    print("\talfa = {}, beta = {}, alfa - beta = {}".format(alfa, beta, alfa - beta))
    y1 = (alfa + beta)
    y2 = (-(a+b)/2 + (1j)*(a-b)*(3**0.5)/2)
    y3 = (-(a+b)/2 - (1j)*(a-b)*(3**0.5)/2)
    return y1, y2, y3


Coef = namedtuple('Coef', 'a b c d')


def get_coefs(mu):

    # t - means sign of calculation (+-)
    tmp_calc = lambda t: t * (mu*mu - 0.16)**0.5

    a = lambda: 1
    b = lambda t: -( 0.5 * tmp_calc(t) - mu / 2 + 0.2)
    c = lambda t: 2.6 * tmp_calc(t)  + 2.4 * mu + 1
    d = lambda t: tmp_calc(t)

    # coef1 = Coef(a(), b(1), c(1), d(1))
    # coef2 = Coef(a(), b(-1), c(-1), d(-1))

    # print("coef1 = {}".format(coef1))
    # print("coef2 = {}".format(coef2))

    return (Coef(a(), b(1), c(1), d(1)), Coef(a(), b(-1), c(-1), d(-1)))


#for mu in np.arange(3.217973511597760, 3.217973511597761, 0.00000000000000001):
for mu in np.arange(2.6, 6.0, 0.05):
    print("-" * 40)
    print("Mu = {}".format(mu))
    for coef in get_coefs(mu):
        print("\tCoef = {}".format(coef))
        print("\tRoots = {}".format(get_roots(coef.a, coef.b, coef.c, coef.d)))
