import os
os.environ['DDE_BACKEND'] = 'tensorflow'

import matplotlib.animation as animation
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib import cm
import random

import tensorflow as tf
import numpy as np
from math import sqrt

!pip install deepxde -q
import deepxde as dde


# setting seed to a fixed value
def set_seed(seed_value = 0):
  np.random.seed(seed_value)
  tf.random.set_seed(seed_value)
  random.seed(seed_value)
  dde.config.set_random_seed(seed_value)
set_seed()


# setting equation and geometry
equation = "mkdv"             # "mkdv" or "kdv"
xmin, xmax = [-10, 1], [10, 2]
tmax = 15
geom = dde.geometry.geometry_2d.Rectangle(xmin=xmin, xmax=xmax)
timeDomain = dde.geometry.TimeDomain(0, tmax)
geomtime = dde.geometry.GeometryXTime(geom, timeDomain)


# defining the equation
def pde(x, y):
    dy_t = dde.grad.jacobian(y, x, i=0, j=2)
    dy_x = dde.grad.jacobian(y, x, i=0, j=0)
    dy_xxx = dde.grad.hessian(dy_x, x, i=0, j=0)

    if equation == "kdv":
      pde = dy_t + y * dy_x + dy_xxx
      return pde

    if equation == "mkdv":
      pde = dy_t + y**2 * dy_x + dy_xxx
      return pde


# pde solution
def pde_sol(X):
  x_arr = X[:, 0:1]
  a_arr = X[:, 1:2]
  t_arr = X[:, 2:3]

  if equation == "kdv":
    c = a_arr / 3
    k = np.sqrt(a_arr / 12)
    return a_arr*np.power(np.cosh(k*(x_arr - c*t_arr)), -2)

  if equation == "mkdv":
    c = a_arr**2 / 6
    k = a_arr / sqrt(6)
    return a_arr*np.power(np.cosh(k*(x_arr - c*t_arr)), -1)


# initial and boundary conditions
def on_boundary_X(x, on_boundary):
  return on_boundary and (dde.utils.isclose(x[0], xmin[0]) or dde.utils.isclose(x[0], xmax[0]))

ic = dde.icbc.IC(geomtime, pde_sol, lambda _, on_initial: on_initial)
bc = dde.icbc.DirichletBC(geomtime, pde_sol, on_boundary_X)    # exact solution at boundary


domain_points = 10000
boundary_points = 4000
initial_points = 4000
data = dde.data.TimePDE(
                        geomtime,
                        pde,
                        [bc, ic],
                        num_domain = domain_points,
                        num_boundary = boundary_points,
                        num_initial = initial_points
                        )


neurons = 60
layers = 5
net = dde.nn.FNN([3] + [neurons] * layers + [1], "tanh", "zeros")
for d in net.denses:
    d.kernel_initializer = tf.keras.initializers.GlorotNormal()
model = dde.Model(data, net)


def train(iterations):
    try:
      os.makedirs(f"model/cpkt")
    except FileExistsError:
      pass

    display_every = 1
    resampler = dde.callbacks.PDEPointResampler(period=1000)
    return model.train(iterations=iterations,
                      display_every=display_every,
                      callbacks=[resampler])


x_num = 200
x = np.linspace(xmin[0], xmax[0], x_num)
def get_arr_X(t, a):
  t_array = np.full_like(x, t)
  a_array = np.full_like(x, a)
  X = np.stack((x, a_array, t_array), axis=1)
  return X


def dde_l2relerror(t, a):
    X = get_arr_X(t, a)
    y_pred = model.predict(X)
    y_true = pde_sol(X)
    return dde.metrics.l2_relative_error(y_true, y_pred)
  
vdde_l2relerror = np.vectorize(dde_l2relerror)


error_data = dict()
def error_surface(grid_size=50):
  fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
  ax.cla()
  ax.set_xlim(xmin[1], xmax[1])
  ax.set_ylim(0, tmax)
  ax.set_xlabel("a")
  ax.set_ylabel("t")
  ax.set_zlabel("Error")
  plt.title("L2 Relative Error")

  A = np.linspace(xmin[1], xmax[1], grid_size)
  T = np.linspace(0, tmax, grid_size)
  A, T = np.meshgrid(A, T)
  Z = vdde_l2relerror(t=T, a=A)
  dde_l2relerror_max = np.max(Z)
  dde_l2relerror_mean = np.mean(Z)
  error_data["dde_l2relerror_max"] = dde_l2relerror_max
  error_data["dde_l2relerror_mean"] = dde_l2relerror_mean

  surf = ax.plot_surface(A, T, Z, cmap=cm.inferno, linewidth=0, antialiased=True)
  fig.colorbar(surf, shrink=0.5, aspect=15, ticks=np.linspace(0,np.max(Z),10), pad=0.1, boundaries=np.linspace(0,np.max(Z),100))

  plt.savefig(f"model/error_surface.png")


def gif(a):
    Figure = plt.figure()
    predicted_line = plt.plot([], 'b-')[0]
    solution_line = plt.plot([], 'r--')[0]
    plt.xlim(xmin[0], xmax[0])
    plt.ylim(-0.5, 2.5)
    frames=120
    def AnimationFunction(frame):
        K= (tmax)/frames
        t = K*frame
        plt.title(f"t={t:.2f}")
        X = get_arr_X(t=t, a=a)
        y_pred = model.predict(X)
        y_sol = pde_sol(X)
        predicted_line.set_data((x, y_pred))
        solution_line.set_data((x, y_sol))

    anim_created = FuncAnimation(Figure, AnimationFunction, frames=frames, interval=25)
    writer = animation.PillowWriter(fps=24,
                                    metadata=dict(artist='Me'),
                                    bitrate=1800)
    anim_created.save(f'model/prediction_vs_exact_amplitude{a}.gif', writer=writer)
    plt.close()



# TRAINING
set_seed()

model.compile("adam", lr=1e-2)
losshistory, train_state = train(iterations=3_000)
model.compile("adam", lr=5e-3)
losshistory, train_state = train(iterations=3_000)
model.compile("adam", lr=1e-3)
losshistory, train_state = train(iterations=4_000)
model.compile("adam", lr=7e-4)
losshistory, train_state = train(iterations=5_000)
model.compile("adam", lr=5e-4)
losshistory, train_state = train(iterations=5_000)
model.compile("adam", lr=3e-4)
losshistory, train_state = train(iterations=5_000)
model.compile("adam", lr=1e-4)
losshistory, train_state = train(iterations=15_000)
model.compile("adam", lr=7e-5)
losshistory, train_state = train(iterations=15_000)
model.compile("adam", lr=5e-5)
losshistory, train_state = train(iterations=15_000)
model.compile("adam", lr=3e-5)
losshistory, train_state = train(iterations=15_000)
model.compile("adam", lr=1e-5)
train(iterations=15_000)
dde.saveplot(losshistory, train_state, issave=True, isplot=True)
model.save("model/ckpt")

error_surface(grid_size=50)
gif(a=2)
