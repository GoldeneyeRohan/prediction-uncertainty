import numpy as np
import scipy
from abc import ABC, abstractmethod

def linear_regression(X, Y, N, M, lamb):
    Q = np.linalg.inv(np.dot(X.T, X) + lamb * np.eye(X.shape[1]))
    b = np.dot(X.T, Y)
    W = np.dot(Q, b)
    A = W.T[:, 0:N]
    B = W.T[:, N:N+M]

    errors = np.dot(X, W) - Y

    return A, B, Q, errors

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

def weighted_least_squares(X, Y, K, n, m, lamb):
    X = np.hstack((X, np.ones((X.shape[0], 1))))
    O = np.diag(K)
    Q = np.linalg.inv(X.T @ O @ X + lamb * np.eye(X.shape[1]))
    W = Q @ X.T @ O @ Y

    A = W.T[:,:n]
    B = W.T[:,n:n+m]
    C = W.T[:,-1]

    errors = X @ W - Y
    return A, B, C, Q, errors

def epanechnikov(x, x_bar, h):
    dists = np.linalg.norm(x - x_bar, axis=1)
    return np.maximum((1 - (dists / h) ** 2) * (3 / 4), 0)

def get_dataset(x_trajs, u_trajs):
    xs, ys = zip(*[(np.hstack((x[:-1], u)), x[1:]) for x, u in zip(x_trajs, u_trajs)])
    x_data = np.vstack(xs)
    y_data = np.vstack(ys)
    return x_data, y_data

def select_points(x_data, y_data, x_bar, n_points):
    dists = np.linalg.norm(x_data - x_bar, axis=1)
    idx = np.argsort(dists)[:n_points]
    return x_data[idx,:], y_data[idx,:]

class SystemID(ABC):

    def __init__(self, n_states, n_inputs):
        self.x_traj_list = []
        self.u_traj_list = []
        self.n_states = n_states
        self.n_inputs = n_inputs

class LocalLinearModel(SystemID):
    def __init__(self, n_states, n_inputs, h, lamb, n_sysid_pts=None, n_sysid_it=None):
        super(LocalLinearModel, self).__init__(n_states, n_inputs)
        self.h = h
        self.lamb = lamb
        self.n_sysid_pts = n_sysid_pts
        self.n_sysid_it = n_sysid_it
        self.X = None
        self.Y = None

    def add_trajectory(self, x_traj, u_traj):
        self.x_traj_list.append(x_traj)
        self.u_traj_list.append(u_traj)

        if self.n_sysid_it is not None:
            self.X, self.Y = get_dataset(self.x_traj_list[-self.n_sysid_it:], self.u_traj_list[-self.n_sysid_it:])
        else: 
            self.X, self.Y = get_dataset(self.x_traj_list, self.u_traj_list)

    def regress_model(self, x_bar, u_bar):
        z_bar = np.hstack((x_bar, u_bar))

        if self.n_sysid_pts is not None:
            x_data, y_data = select_points(self.X, self.Y, z_bar, self.n_sysid_pts)
        else:
            x_data = self.X
            y_data = self.Y

        K = epanechnikov(x_data, z_bar, self.h)

        A, B, C, Q, errors = weighted_least_squares(x_data, y_data, K, self.n_states, self.n_inputs, self.lamb)
        return A, B, C, Q, errors

    def predict(self, x_test, u_test):
        x_preds = []
        covs = []
        for x, u in zip(np.rollaxis(x_test, 0), np.rollaxis(u_test, 0)):
            z = np.hstack([x, u, 1])
            A, B, C, Q, errors = self.regress_model(x, u)
            x_pred = A @ x + B @ u + C
            cov = z.T @ Q @ z
            x_preds.append(x_pred)
            covs.append(cov)
        return np.vstack(x_preds), np.array(covs)

    def regress_models(self, x_traj, u_traj):
        As, Bs, Cs, covs, errors = zip(*[self.regress_model(x_traj[i,:], u_traj[i,:]) for i in range(u_traj.shape[0])])
        return As, Bs, Cs, covs, errors

