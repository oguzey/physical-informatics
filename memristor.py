import numpy as np


class Memristor:
    def __init__(self, D=None, mu=None, nu=None, w0=None, R_ON=None, R_OFF=None, U=None, Phi=None):
        self.__D = D
        self.__mu = mu
        self.__nu = nu
        self.__w0 = w0
        self.__R_ON = R_ON
        self.__R_OFF = R_OFF

        self.U = U
        self.Phi = Phi

        self.__w_initial = (
            (self.__w0**2) * (self.__R_ON - self.__R_OFF) / (2 * self.__D) + self.__w0 * self.__R_OFF
        )

    def w(self, t):
        up_sqrt = (
            self.__R_OFF**2 + 2 * ((self.__R_ON - self.__R_OFF) / self.__D) *
            (self.__w_initial + self.__mu * (self.__R_ON / self.__D) * self.Phi(t))
        )
        up = up_sqrt**.5 - self.__R_OFF
        values = self.__D * up / (self.__R_ON - self.__R_OFF)
        values[values >= self.__D] = self.__D
        values[values <= 0] = 0
        return values

    def I(self, t):
        return (self.U(t) /
                (self.__R_ON * self.w(t) / self.__D + self.__R_OFF * (1 - self.w(t) / self.__D))
                )

    def q(self, T):
        result = [0]
        last_t = T[0]
        I = self.I(T)
        for t, i in zip(T[1:], I[:-1]):
            result.append(result[-1] + i * (t - last_t))
            last_t = t
        return result[1:]

    @staticmethod
    def __make_figure(plt_l, name):
        fig = plt_l.figure()
        fig.set_size_inches(15, 7)
        fig.canvas.set_window_title(name.capitalize())

    def task1(self, plt, N=None, periods=None, need_U_t=True):
        T = np.pi * periods / self.__nu
        TIME = np.linspace(0, T, N)

        f, axarr = plt.subplots(3 if need_U_t else 2, sharex=True)
        f.canvas.set_window_title("Task1")

        axarr[0].set_title(r'$\frac{\omega\left( t \right)}{D}$')
        axarr[0].plot(TIME, self.w(TIME) / self.__D)
        axarr[0].grid(b=True, which='major', color='grey', linestyle='--')
        axarr[0].set_ylabel(r'$\frac{\omega}{D}$')

        axarr[1].set_title(r'$I\left( t \right)$')
        axarr[1].plot(TIME, self.I(TIME))
        axarr[1].grid(b=True, which='major', color='grey', linestyle='--')
        axarr[1].set_ylabel('$I$')
        if need_U_t:
            axarr[2].set_title(r'$U\left( t \right)$')
            axarr[2].plot(TIME, self.U(TIME))
            axarr[2].grid(b=True, which='major', color='grey', linestyle='--')
            axarr[2].set_ylabel('$U$')

        plt.xlabel('$t$')

    def task2(self, plt, N=None, periods=None):
        T = np.pi * periods / self.__nu
        TIME = np.linspace(0, T, N)

        self.__make_figure(plt, "task2")
        plt.plot(self.U(TIME), self.I(TIME))
        plt.grid(b=True, which='major', color='grey', linestyle='--')
        plt.title(r'$I\left( U \right)$')
        plt.xlabel('$U$')
        plt.ylabel('$I$')

    def task3(self, plt, N=None, periods=None):
        T = np.pi * periods / self.__nu
        TIME = np.linspace(0, T, N)

        self.__make_figure(plt, "task3")
        plt.plot(self.Phi(TIME)[1:], self.q(TIME))
        plt.grid(b=True, which='major', color='grey', linestyle='--')
        plt.title(r'$q\left( \Phi \right)$')
        plt.xlabel('$\Phi$')
        plt.ylabel('$q$')
