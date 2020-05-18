import numpy as np
import scipy
from abc import ABC, abstractmethod

def regression(X, Y, N, M, lamb):
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

def SystemID(ABC):

    def __init__(self):

    @abstractmethod
    def fit(self):
        raise(NotImplementedError)



