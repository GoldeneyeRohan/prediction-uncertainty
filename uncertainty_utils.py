import numpy as np
import controlpy
import matplotlib.pyplot as plt
from matplotlib import animation

def regression(x, u, lamb):
    """Estimates linear system dynamics
    x, u: date used in the regression
    lamb: regularization coefficient
    """

    # Want to solve W^* = argmin sum_i ||W^T z_i - y_i ||_2^2 + lamb ||W||_F,
    # with z_i = [x_i u_i] and W \in R^{n + d} x n
    N = x.shape[1]
    M = u.shape[1]

    Y = x[1:x.shape[0], :]
    X = np.hstack((x[0:(x.shape[0] - 1), :], u[0:(x.shape[0] - 1), :]))

    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)
    A = W.T[:, 0:N]
    B = W.T[:, N:N+M]

    ErrorMatrix = np.dot(X, W) - Y

    return A, B, Q, ErrorMatrix

class KF:

    def __init__(self, P, process_noise, measurement_noise, A, B, C, x_init):
        self.P = P
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise
        self.A = A
        self.B = B
        self.C = C
        self.x = x_init

    def update_estimate(self, y, u):
        # predict
        x_priori = self.A @ self.x + self.B @ u
        P_priori = self.A @ self.P @ self.A.T + self.process_noise

        # update
        S = self.C @ P_priori @ self.C.T + self.measurement_noise
        K = P_priori @ self.C.T @ np.linalg.inv(S)
        P_post = (np.eye(self.P.shape[0]) - K @ self.C) @ P_priori
        x_post = x_priori + K @ (y - self.C @ x_priori)

        #store results
        self.x = x_post
        self.P = P_post

        return self.x

def sim_traj(A, B, K, Q, R, process_noise, x_init, N=50, input_limits=np.array([-1e9, 1e9])):
    x_traj = [x_init]
    u_traj = []

    h = lambda x,u: x.T @ Q @ x + u.T @ R @ u
    est_cost = 0
    cost = 0

    for _ in range(N):
        u = np.minimum(np.maximum(input_limits[0], - K @ x_traj[-1]), input_limits[1])
        x_next = A @ x_traj[-1] + B @ u + np.random.multivariate_normal(np.zeros(x_init.shape[0]), process_noise)
        cost += h(x_traj[-1], u)

        x_traj.append(x_next)
        u_traj.append(u)
    return np.array(x_traj), np.array(u_traj), cost

def plot_covariance(K, cov, sigma, xlim, ylim, n_grid = 100):
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    XX = np.vstack((X.flatten(),Y.flatten()))
    U =  - K @ XX
    XU = np.vstack((XX, U))
    uncerts = np.array([sigma * XU[:,i].T @ cov @ XU[:,i] for i in range(XU.shape[1])]).reshape(X.shape)
    return X, Y, uncerts

def animate_single_trajectories(data, file):
    fig = plt.figure(figsize=(8,8))
    xlim = (-3, 15)
    ylim = (-8, 5)
    ax = plt.axes(xlim=xlim, ylim=ylim)
    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    line1, = ax.plot([], [], "b.", label="data")
    line2, = ax.plot([], [], "r", label="trajectory")
    X, Y, uncert = plot_covariance(data["K"][0], data["covariance"][0], data["process noise"], xlim, ylim)
    line3 = ax.contourf(X, Y, uncert)
    cb = fig.colorbar(line3)
    ax.legend(loc="upper right")

    # initialization function: plot the background of each frame
    def init():
        line1.set_data([], [])
        line2.set_data([], [])
        return line1,line2,

    # animation function.  This is called sequentially
    def animate(i):
        line1.set_data(data["state traj"][i][:,0], data["state traj"][i][:,1])
        line2.set_data(data["state traj"][i-1][:,0], data["state traj"][i-1][:,1])
        X, Y, uncert = plot_covariance(data["K"][i], data["covariance"][i], data["process noise"], xlim, ylim)
        # line3.set_data(X, Y, uncert)
        line3 = ax.contourf(X,Y, uncert)
        cb.update_normal(uncert)
        ax.set_title("trajectory {}".format(i))
        return line1, line2

    # call the animator.  blit=True means only re-draw the parts that have changed.
    anim = animation.FuncAnimation(fig, animate, init_func=init,
                                   frames=len(data["state traj"]), blit=True)
    anim.save(file, fps=5, extra_args=['-vcodec', 'libx264'])






