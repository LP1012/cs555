import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.sparse import csr_matrix, identity
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
        case "bfd1":
            return bfd1(u_prev1, dt, A, nu)
        case "bfd2":
            return bfd2(u_prev1, u_prev2, dt, A, nu)
        case "cn":
            return cn(u_prev1, dt, A, nu)
        case _:
            raise ValueError(
                f"time_stepper value not valid! time_stepper = {time_stepper}"
            )


def bfd1(u_prev, dt, A, nu):
    n = len(u_prev)
    mat = identity(n) + dt * nu * A
    u_next = spsolve(mat, u_prev)
    return u_next


def bfd2(u_prev1, u_prev2, dt, A, nu):
    n = len(u_prev1)
    mat = 3 * identity(n) + 2 * nu * dt * A
    vec = 4 * u_prev1 - u_prev2
    u_next = spsolve(mat, vec)
    return u_next


def cn(u_prev, dt, A, nu):
    n = len(u_prev)
    coeff = dt / 2 * nu
    w = (identity(n) - coeff * A) @ u_prev
    mat = identity(n) + coeff * A
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
nx = 15000
n_timesteps = [25, 50, 100, 200, 400, 800, 1600]

time_stepper_index = 1  # 1=bfd1, 2=bfd2, 3=Crank-Nicholson
ic_index = 1  # value of 1 or 2
n_curves = 5  # number of curves in the plot


# ----------------------------------------------------------
# numerical solution
# ----------------------------------------------------------

xs, dx = np.linspace(0, length, nx, retstep=True)

time_steppers = ["bfd1", "bfd2", "cn"]
time_stepper = time_steppers[time_stepper_index - 1]

n = nx - 2  # homogeneous dirichlet conditions, so we don't need to solve for them
A = create_space_fd_mat(n, dx)

lam_ex = -nu * np.pi**2 / length

# initialize output table
df = pd.DataFrame(
    {"Time Differencing": [], "nx": [], "nt": [], "Relative L2 Error": [], "Ratio": []}
)

prev_error = None
for time_steps in n_timesteps:
    plt.figure()
    plt.xlabel("x")
    plt.ylabel("u(x)")
    plt.title(
        f"Solution to the 1D, time-dependent heat equation\ntime-stepper={time_stepper}, n_steps = {time_steps}"
    )
    n_between_plots = round(time_steps / (n_curves))
    dt = t_max / time_steps
    u_prev = np.zeros(n)
    u_current = np.array([ic(ic_index, x) for x in xs])
    plt.plot(xs, u_current, label="t = 0")
    u_current = u_current[1:-1]

    for t in range(1, time_steps + 1):
        if time_stepper == "bfd2" and t == 1:
            # bootstrap if necessary
            u_next = timeStep(u_current, u_prev, dt, A, nu, "bfd1")
        else:
            u_next = timeStep(u_current, u_prev, dt, A, nu, time_stepper)
        if t == time_steps or t % n_between_plots == 0:
            prepend = np.insert(u_next, 0, 0)
            u_plot = np.append(prepend, 0)
            plt.plot(xs, u_plot, label=f"t = {t}")

        t_current = t * dt
        u_exact = np.array(
            [np.exp(t_current * lam_ex) * np.sin(np.pi * x) for x in xs[1:-1]]
        )
        error_vec = u_exact - u_next

        el2 = np.sqrt(np.dot(error_vec, error_vec) / np.dot(u_exact, u_exact))

        u_prev = u_current
        u_current = u_next

    if prev_error is not None:
        ratio = prev_error / el2
    else:
        ratio = None  # first run has no convergence ratio

    prev_error = el2

    new_table_row = [
        time_stepper,
        nx,
        time_steps,
        el2,
        ratio if ratio is not None else "None",
    ]
    df.loc[len(df)] = new_table_row

    plt.legend()
    plt.savefig(f"p1_{time_stepper}_{time_steps}.png")

df.to_csv(f"p1_table_{time_stepper}.csv")
