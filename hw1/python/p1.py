import numpy as np
import numpy.linalg


def create_space_fd_mat(n, dx):
    diag = np.ones(n) * 2
    bands = -np.ones(n - 1)
    A = np.diag(diag) + np.diag(bands, k=-1) + np.diag(bands, k=1)
    return A * (1 / dx**2)


def timeStep(u_prev1, dt, A, nu, time_stepper, u_prev2=None):
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
nx = 100
n_timesteps = [10, 100]
time_stepper_index = 1
ic_index = 1

# begin code...
time_steppers = ["bdf1", "bdf2", "cn"]
time_stepper = time_steppers[time_stepper_index]

dx = length / nx
