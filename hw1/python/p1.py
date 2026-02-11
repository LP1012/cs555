import numpy as np
import numpy.linalg
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, csr_array
from scipy.sparse.linalg import spsolve

plt.style.use("matplotlib.rc")


def create_space_fd_mat(n, dx):
    diag = np.ones(n) * 2
    bands = -np.ones(n - 1)
    A = np.diag(diag) + np.diag(bands, k=-1) + np.diag(bands, k=1)
    dense_mat = A * (1 / dx**2)
    return csr_matrix(dense_mat)


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
    mat = csr_matrix(mat)
    u_next = spsolve(mat, u_prev)
    return u_next


def bdf2(u_prev1, u_prev2, dt, A, nu):
    n = len(u_prev1)
    mat = 3 * np.eye(n) + 2 * nu * dt * A
    mat = csr_matrix(mat)
    vec = 4 * u_prev1 - u_prev2
    u_next = spsolve(mat, vec)
    return u_next


def cn(u_prev, dt, A, nu):
    n = len(u_prev)
    coeff = dt / 2 * nu
    w = (np.eye(n) - coeff * A) @ u_prev
    mat = np.eye(n) + coeff * A
    mat = csr_matrix(mat)
    u_next = spsolve(mat, w)
    return u_next


def ic(index, x):
    match index:
        case 1:
            return np.sin(np.pi * x)
        case 2:
            return np.pi / 4
        case _:
            raise ValueError(f"Index number passed not valid! index = {index}")


# ----------------------------------------------------------
# user-set parameters
# ----------------------------------------------------------
length = 1
nu = 0.2
t_max = 1
nx = 1500
n_timesteps = [25, 50, 100, 200, 400, 800, 1600]

time_stepper_index = 2  # 1=bdf1, 2=bdf2, 3=Crank-Nicholson
ic_index = 1  # value of 1 or 2
n_curves = 4  # number of curves in the plot


# ----------------------------------------------------------
# numerical solution
# ----------------------------------------------------------

xs, dx = np.linspace(0, length, nx, retstep=True)

time_steppers = ["bdf1", "bdf2", "cn"]
time_stepper = time_steppers[time_stepper_index - 1]

n = nx - 2  # homogeneous dirichlet conditions, so we don't need to solve for them
A = create_space_fd_mat(n, dx)

lam_ex = -nu * np.pi**2 / length

# initialize output table
df = pd.DataFrame(
    {"Time Differencing": [], "nx": [], "nt": [], "Final Error": [], "Ratio": []}
)


for time_steps in n_timesteps:
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(
        f"Solution to the 1D, time-dependent heat equation\ntime-stepper={time_stepper}, n_steps = {time_steps}"
    )
    n_between_plots = round(time_steps / (n_curves - 1))
    dt = t_max / time_steps
    u_prev = np.zeros(n)
    u_current = np.array([ic(ic_index, x) for x in xs])
    plt.plot(xs, u_current, label="t = 0")
    u_current = u_current[1:-1]

    old_error = None
    new_error = None
    for t in range(1, time_steps + 1):
        if time_stepper == "bdf2" and t == 1:
            # bootstrap if necessary
            u_next = timeStep(u_current, u_prev, dt, A, nu, "bdf1")
        else:
            u_next = timeStep(u_current, u_prev, dt, A, nu, time_stepper)
        if t == time_steps - 1 or t % n_between_plots == 0:
            prepend = np.insert(u_next, 0, 0)
            u_plot = np.append(prepend, 0)
            plt.plot(xs, u_plot, label=f"t = {t+1}")

        t_current = t * dt
        u_exact = np.array(
            [np.exp(t_current * lam_ex) * np.sin(np.pi * x) for x in xs[1:-1]]
        )
        error_vec = u_exact - u_next
        # ratio = np.max(np.abs(error_vec)) / np.max(np.abs(u_exact))

        old_error = new_error  # reassign
        new_error = np.sqrt(
            np.dot(error_vec, error_vec) / np.dot(u_exact, u_exact)
        )  # new values

        u_prev = u_current
        u_current = u_next

    ratio = old_error / new_error

    new_table_row = [time_stepper, nx, time_steps, new_error, ratio]
    df.loc[len(df)] = new_table_row

    plt.legend()
    plt.savefig(f"p1_{time_stepper}_{time_steps}.png")

df.to_csv(f"p1_table_{time_stepper}.csv")
