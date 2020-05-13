import numpy as np
import controlpy
import scipy
import matplotlib.pyplot as plt
from matplotlib import animation

THETAS = np.linspace(0, 2 * np.pi)
Z_UNIT = np.vstack((np.cos(THETAS), np.sin(THETAS)))

def regression(X, Y, N, M, lamb):


    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)
    A = W.T[:, 0:N]
    B = W.T[:, N:N+M]

    ErrorMatrix = np.dot(X, W) - Y

    return A, B, Q, ErrorMatrix

def single_traj_regression(x, u, lamb):
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
    return regression(X,Y, N, M, lamb)
    
def multi_traj_regression(x_trajs, u_trajs, lamb):
    N = x_trajs[0].shape[1]
    M = u_trajs[0].shape[1]
    
    xs, ys = zip(*[(np.hstack((x[0:-1,:], u[:x.shape[0]-1,:])), x[1:,:]) for x, u in zip(x_trajs, u_trajs)])
    X = np.vstack(xs)
    Y = np.vstack(ys)
    return regression(X,Y, N, M, lamb)

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

def sim_pred_traj(A_est, B_est, x_init, u_traj):
    x_pred_traj = [x_init]
    for u in np.rollaxis(u_traj ,0):
        x_pred = A_est @ x_pred_traj[-1] + B_est @ u
        x_pred_traj.append(x_pred)
    return np.vstack(x_pred_traj)

def plot_covariance(K, cov, var, xlim, ylim, n_grid = 100):
    x = np.linspace(xlim[0], xlim[1], n_grid)
    y = np.linspace(ylim[0], ylim[1], n_grid)
    X, Y = np.meshgrid(x, y)
    XX = np.vstack((X.flatten(),Y.flatten()))
    U =  - K @ XX
    XU = np.vstack((XX, U))
    uncerts = np.array([var * XU[:,i].T @ cov @ XU[:,i] for i in range(XU.shape[1])]).reshape(X.shape)
    return X, Y, uncerts

def calc_t(confidence, dimension):
    pi_power = (2 * np.pi) ** (dimension / 2 - 1)
    alpha = (pi_power - confidence) / pi_power
    t = np.sqrt(2) * np.sqrt(- np.log(alpha))
    return t

def calc_t_chebyshev(confidence, dimension):
    t = np.sqrt(dimension / (1 - confidence))
    return t

def calc_covariances_input_traj(expected_state_traj, input_traj, process_noise, regression_cov, A_est):
    z_vecs = np.hstack((expected_state_traj[:-1,:], input_traj))
    # First state is known
    n = expected_state_traj.shape[1]
    d = input_traj.shape[1]
    covs = [np.zeros((n, n))]
    # Recurse for covariances
    for z in np.rollaxis(z_vecs, 0):
        z_cov = np.zeros((n + d, n + d))
        z_cov[:n,:n] = covs[-1]
        r = (1 + z.T @ regression_cov @ z + np.trace(regression_cov @ z_cov)) 
        cov = process_noise * r +  A_est @ covs[-1] @ A_est.T
        covs.append(cov)
    return covs

def calc_covariances_state_feedback(A_est, B_est, K, process_noise, regression_cov, x0, T):
    A_clp = A_est - B_est @ K
    x_sim = [np.linalg.matrix_power(A_clp, i) @ x0 for i in range(T)]
    x_sim = np.vstack(x_sim)

    n = A_est.shape[0]
    Q = regression_cov[:n, :n]
    S = regression_cov[:n, n:]
    R = regression_cov[n:, n:]
    P = Q - S @ K - K.T @ S.T + K.T @ R @ K

    # Recurse for covariances
    covs = [np.zeros(A_est.shape)]
    for x in np.rollaxis(x_sim, 0):
        r = (1 + x.T @ P @ x + np.trace(P @ covs[-1])) 
        cov = process_noise * r +  A_clp @ covs[-1] @ A_clp.T
        covs.append(cov)

    return covs

def mp(x, i):
    return np.linalg.matrix_power(x,i)

def sum_func(f, idx):
    return np.sum([f(i) for i in idx], axis=0)

def qf(x, P):
    return x.T @ P @ x

def calc_covariances_state_feedback_explicit(A_est, B_est, K, process_noise, regression_cov, x0, T):
    A_clp = A_est - B_est @ K
    x_sim = [np.linalg.matrix_power(A_clp, i) @ x0 for i in range(T)]
    A_S = lambda i: mp(A_clp, i) @ process_noise @ mp(A_clp, i).T

    n = A_est.shape[0]
    Q = regression_cov[:n, :n]
    S = regression_cov[:n, n:]
    R = regression_cov[n:, n:]
    P = Q - S @ K - K.T @ S.T + K.T @ R @ K

    covs = []
    for k, x in enumerate(x_sim):
        if k > 1:
            cov_nominal = sum_func(lambda i : A_S(i), range(k))
            f = lambda k, i, j: sum_func(lambda l: np.trace(P @ A_S(j)) ** l, range(1, k - j - i))
            g = lambda k, i: sum_func(lambda j: f(k, i, j), range(k - i - 1))
            h = lambda k: sum_func(lambda i: A_S(i) * g(k, i), range(k - 1))
            cov_constants = cov_nominal + h(k)

            cov_state_nominal = sum_func(lambda i: qf(x_sim[i], P) * A_S(k - i - 1), range(k))
            f_s = lambda k, i, j: sum_func(lambda l: np.trace(P @ A_S(l)) ** (k - j - l - i - 1),range(k - i - j - 1))
            g_s = lambda k, i: sum_func(lambda j: A_S(j) * f_s(k, i, j), range(k - i - 1))
            h_s = lambda k: sum_func(lambda i: qf(x_sim[i], P) * g_s(k, i), range(k - 1))
            cov_state = cov_state_nominal + h_s(k)

            cov = cov_state + cov_constants
            covs.append(cov)
    return covs 

def get_elipse(t, cov, mean):
    zs = Z_UNIT * t
    xs = scipy.linalg.sqrtm(cov) @ zs
    xs = xs.T + mean
    return xs

def get_elipses(t, covs, means):
    ellipses = []
    for cov, mean in zip(covs, np.rollaxis(means,0)):
        ellipse_data = get_elipse(t, cov, mean)
        ellipses.append(ellipse_data)
    return ellipses

def animate_single_trajectories(data, file):
    fig = plt.figure(figsize=(8,8))
    xlim = (-3, 15)
    ylim = (-8, 5)
    def animate(i):
        plt.clf()
        if i > 0:
            plt.plot(data["state traj"][i-1][:,0], data["state traj"][i-1][:,1], "k.", label="data")
        plt.plot(data["state traj"][i][:,0], data["state traj"][i][:,1], "ro-", label="trajectory")
        X, Y, uncert = plot_covariance(data["K"][i], data["covariance"][i], data["process noise variance"], xlim, ylim)
        cont = plt.contourf(X,Y, uncert, cmap="bwr", vmin=0, vmax=0.6)
        m = plt.cm.ScalarMappable(cmap="bwr")
        m.set_array(uncert)
        m.set_clim(0., .6)
        plt.colorbar(m)
        plt.title("trajectory {}".format(i))
        plt.legend(loc="upper right")
        plt.xlabel("x1")
        plt.ylabel("x2")
        return

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(data["state traj"]), blit=False)
    anim.save(file, fps=5, extra_args=['-vcodec', 'libx264'])


def animate_multi_trajectories(data, file):
    fig = plt.figure(figsize=(8,8))
    xlim = (-3, 15)
    ylim = (-8, 5)
    def animate(i):
        plt.clf()
        if i > 0:
            state_data = np.vstack(data["state traj"][:i])
            plt.plot(state_data[:,0], state_data[:,1], "k.", label="data")
        plt.plot(data["state traj"][i][:,0], data["state traj"][i-1][:,1], "ro-", label="trajectory")
        X, Y, uncert = plot_covariance(data["K"][i], data["covariance"][i], data["process noise variance"], xlim, ylim)
        cont = plt.contourf(X,Y, uncert, cmap="bwr")
        plt.colorbar()
        plt.title("trajectory {}".format(i))
        plt.legend(loc="upper right")
        plt.xlabel("x1")
        plt.ylabel("x2")
        return

    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(data["state traj"]), blit=False)
    anim.save(file, fps=5, extra_args=['-vcodec', 'libx264'])

def animate_confidence_bounds(data, file, A_true, B_true, confidence, process_noise, xlim=(-3, 15), ylim =(-8, 5)):
    fig = plt.figure(figsize=(16,8))

    t = calc_t(confidence, A_true.shape[0])
    t_cheby = calc_t_chebyshev(confidence, A_true.shape[0])
    names = ["True VS Estimated Uncertainty for episode ", "Estimated VS Approximated Uncertainty for episode "]
    def animate(j):
        fig.clf()
        axes = None
        axes = [fig.add_subplot(121), fig.add_subplot(122)]
        x_traj = data["state traj"][j]
        u_traj = data["input trajectory"][j]
        A_est = data["A_est"][j]
        B_est = data["B_est"][j]
        model_cov = data["covariance"][j]
        if j > 0:
            state_data = np.vstack(data["state traj"][:j])
            axes[0].plot(state_data[:,0], state_data[:,1], "k.", label="data")
            axes[1].plot(state_data[:,0], state_data[:,1], "k.", label="data")

        axes[0].plot(x_traj[:,0], x_traj[:,1],"r-o", label="trajectory")
        axes[1].plot(x_traj[:,0], x_traj[:,1],"r-o", label="trajectory")

        for i in range(u_traj.shape[0]):
            x = x_traj[i,:]
            u = u_traj[i,:]
            x_pred_est_mean = A_est @ x + B_est @ u
            x_pred_true_mean = A_true @ x + B_true @ u
            z = np.hstack((x,u))
            est_cov = (1 + z.T @ model_cov @ z) * process_noise
            true_confidence = get_elipse(t, process_noise, x_pred_true_mean)
            est_confidence = get_elipse(t, est_cov, x_pred_est_mean)
            cheby_confidence = get_elipse(t_cheby, est_cov, x_pred_est_mean)
            if i == 0:
                axes[0].plot(true_confidence[:,0], true_confidence[:,1], "b", label="true confidence")
                axes[0].plot(est_confidence[:,0], est_confidence[:,1], "r", label="est confidence")
                axes[1].plot(est_confidence[:,0], est_confidence[:,1], "r", label="est confidence")
                axes[1].plot(cheby_confidence[:,0], cheby_confidence[:,1], "tab:orange", label="chebyshev confidence")
            else:
                axes[0].plot(true_confidence[:,0], true_confidence[:,1], "b")
                axes[0].plot(est_confidence[:,0], est_confidence[:,1], "r")
                axes[1].plot(est_confidence[:,0], est_confidence[:,1], "r")
                axes[1].plot(cheby_confidence[:,0], cheby_confidence[:,1], "tab:orange")


        for anum, ax in enumerate(axes):
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend()
            ax.set_title(names[anum] + str(j))
            ax.set_xlabel("x1")
            ax.set_ylabel("x2")


    anim = animation.FuncAnimation(fig, animate,
                                   frames=len(data["state traj"]), blit=False)
    anim.save(file, fps=5, extra_args=['-vcodec', 'libx264'])




