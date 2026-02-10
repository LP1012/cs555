import numpy as np
import numpy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use("rc-general.mplstyle")


def create_space_fd_mat(n, dx):
    diag = np.ones(n) * 2
    bands = -np.ones(n - 1)
    A = np.diag(diag) + np.diag(bands, k=-1) + np.diag(bands, k=1)
    return A * (1 / dx**2)


def timeStep(u_prev1, u_prev2, dt, A, nu, time_stepper):
    match time_stepper:
        case "bdf1":
            return bdf1(u_prev1, dt, A, nu)
        case "bdf2":
            return bdf2(u_prev1, u_prev2, dt, A, nu)
        case "cn":
            return cn(u_prev1, dt, A, nu)
        case _:
            raise ValueError(
                f"time_stepper value not valid! time_stepper = {time_stepper}"
            )


def bdf1(u_prev, dt, A, nu):
    n = len(u_prev)
    mat = np.eye(n) + dt * nu * A
    u_next = np.linalg.solve(mat, u_prev)
    return u_next


def bdf2(u_prev1, u_prev2, dt, A, nu):
    n = len(u_prev1)
    mat = 3 * np.eye(n) + 2 * dt * A
    vec = 4 * u_prev1 - u_prev2
    u_next = np.linalg.solve(mat, vec)
    return u_next


def cn(u_prev, dt, A, nu):
    n = len(u_prev)
    coeff = dt / 2 * nu
    w = (np.eye(n) - coeff * A) @ u_prev
    mat = np.eye(n) + coeff * A
    u_next = np.linalg.solve(mat, w)
    return u_next


def ic(index, x):
    match index:
        case 1:
            return np.sin(np.pi * x)
        case 2:
            return np.pi / 4
        case _:
            raise ValueError(f"Index number passed not valid! index = {index}")


# user-set parameters
length = 1
nu = 0.2
t_max = 1
nx = 1500
n_timesteps = [25, 50, 100, 200, 400, 800]
time_stepper_index = 1
ic_index = 1
n_curves = 4

# begin code...
time_steppers = ["bdf1", "bdf2", "cn"]
time_stepper = time_steppers[time_stepper_index]

xs, dx = np.linspace(0, length, nx, retstep=True)

n = nx - 2  # homogeneous dirchlet conditions, so we don't need to solve for them
A = create_space_fd_mat(n, dx)

plt.figure()
for time_steps in n_timesteps:
    dt = t_max / time_steps
    u_prev = np.zeros(n)
    u_current = np.array([ic(ic_index, x) for x in xs])
    for t in range(1, time_steps):
        u_next = timeStep(u_current, u_prev, dt, A, nu, time_stepper)
