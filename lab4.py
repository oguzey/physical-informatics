from scipy.integrate import odeint
import matplotlib.pyplot as plt
import numpy as np

def Rosler(state,t):
  # unpack the state vector
  y1 = state[0]
  y2 = state[1]
  y3 = state[2]

  # these are our constants
  alpha = 0.2
  mu = 2.6

  # compute state derivatives
  y1d = - (y2 + y3)
  y2d = y1 + alpha * y2
  y3d = alpha + y3 * (y1 - mu)

  # return the state derivatives
  return [y1d, y2d, y3d]


state0 = [2.0, -1.0, 0.0]
T = 20.0
t = np.arange(0.0, T, 0.01)

state = odeint(Rosler, state0, t)

# do some fancy 3D plotting
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(state[:,0],state[:,1],state[:,2])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
plt.show()
