import matplotlib.pyplot as plt
# do some fancy 3D plotting
from mpl_toolkits.mplot3d import Axes3D


def make_figure(name):
    fig = plt.figure()
    fig.set_size_inches(15, 7)
    fig.canvas.set_window_title(name.capitalize())


def draw_plot_xyz(data_x, data_y, data_z, name, name_x='x', name_y='y', name_z='x'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.set_title(name)
    ax.plot(list(data_x), list(data_y), list(data_z))
    ax.set_xlabel(name_x)
    ax.set_ylabel(name_y)
    ax.set_zlabel(name_z)


def draw_plot_xy(x, y, name, xlabel='x', ylabel='y'):
    make_figure(name)
    plt.title(name)
    plt.plot(x, y, 'o-', label=name)
    plt.grid(b=True, which='major', color='grey', linestyle='--')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)


def show_all_plots():
    plt.show()
