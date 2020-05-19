import numpy as np
import scipy.integrate
from abc import ABC, abstractmethod

GRAVITY = 9.81

class NonLinearSystem(ABC):

    def __init__(self, n_states, n_inputs, init_state, dt, process_noise, use_ode_integrator=True):
        self.x = init_state
        self.n_states = n_states
        self.n_inputs = n_inputs
        self.process_noise = process_noise
        self.use_ode_integrator = use_ode_integrator
        self.t = 0
        self.dt = dt

    @abstractmethod
    def f_continuous(self, state, u):
        """
        Evaluate state derivative
        returns d/dt f(state, u)
        (is a member function because depends on parameters)
        """
        raise(NotImplementedError)

    def f_discrete(self, state, u, dt):
        """
        Calculate transition dynamics
        returns x[t+1] = f_disc(x[t], u[t])
        """
        if self.use_ode_integrator:
            f_zoh = lambda x, t: self.f_continuous(x, u)
            ts = [self.t, self.t + dt]
            states = scipy.integrate.odeint(f_zoh, state, ts)
            return states[-1,:]
        else: 
            return state + self.f_continuous(state, u) * dt

    def f(self, u):
        """
        advances vehicle state by 1 timestep
        """
        self.x = self.f_discrete(self.x, u, self.dt) + np.random.multivariate_normal(np.zeros(self.n_states), self.process_noise)
        self.t += self.dt
        return self.x

    @abstractmethod
    def get_jacobians(self, x_bar, u_bar):
        """
        return system jacobians at x_bar, u_bar
        Jx = df/dx, Ju = df/du
        """
        raise(NotImplementedError)


    def get_linearization(self, x_bar, u_bar, dt):
        """
        Returns a locally linear affine model
        x[t+1] = f(x[t], u[t]) ~~ Ax[t] + Bu[t] + C
        """
        Jx, Ju = self.get_jacobians(x_bar, u_bar)
        A = np.eye(self.n_states) + dt * Jx
        B = dt * Ju
        C = self.f_discrete(x_bar, u_bar, dt) - A @ x_bar - B @ u_bar
        return A, B, C
        

class DubinCar(NonLinearSystem):
    """
    state = [x, y, theta, v], input = [a, delta]
    dx = v*cos(theta)
    dy = v*sin(theta)
    dtheta = delta
    dv = a
    """

    def __init__(self, init_state, dt, process_noise, use_ode_integrator=True):
        super(DubinCar, self).__init__(4, 2, init_state, dt, process_noise, use_ode_integrator=use_ode_integrator)

    def f_continuous(self, state, u):
        x, y, theta, v = state
        alpha, delta = u
        return np.array([v * np.cos(theta), v * np.sin(theta), delta, alpha])

    def get_jacobians(self, x_bar, u_bar):
        x, y, theta, v = x_bar
        a, delta = u_bar

        # Dynamics Jacobian
        Jx = np.zeros((self.n_states, self.n_states))
        # dx/dstate
        Jx[0,2] = - v * np.sin(theta)
        Jx[0,3] = np.cos(theta)
        # dy/dstate
        Jx[1,2] = v * np.cos(theta)
        Jx[1,3] = np.sin(theta)

        # Input Jacobian
        Ju = np.zeros((self.n_states, self.n_inputs))
        Ju[2,1] = 1
        Ju[3,0] = 1

        return Jx, Ju

class PlanarQuadrotor(NonLinearSystem):
    """
    state = [x, y, theta, dx, dy, dtheta], input = [u_f, u_r]
    ddx = - (1/m) * sin(theta) * (uf + ur)
    ddy = (1/m) * cos(theta) * (uf + ur)
    ddtheta = (1/I) (l * uf - l * ur)
    """

    def __init__(self, init_state, m, l, I, dt, process_noise, use_ode_integrator=True):
        super(PlanarQuadrotor, self).__init__(6, 2, init_state, dt, process_noise, use_ode_integrator=use_ode_integrator)
        self.m = m
        self.l = l
        self.I = I

    def f_continuous(self, state, u):
        x, y, theta, dx, dy, dtheta = state
        u_f, u_r = u

        ddx =  - (1 / self.m) * np.sin(theta) * (u_f + u_r)
        ddy = (1 / self.m) * np.cos(theta) * (u_f + u_r) - GRAVITY
        ddtheta = (1 / self.I) * (self.l * u_f - self.l * u_r)
        return np.array([dx, dy, dtheta, ddx, ddy, ddtheta])

    def get_jacobians(self, x_bar, u_bar):
        x, y, theta, dx, dy, dtheta = x_bar
        u_f, u_r = u_bar

        # Dynamics Jacobian
        Jx = np.zeros((self.n_states, self.n_states))
        Jx[0:3,3:] = np.eye(3)
        Jx[3,2] = - (1 / self.m) * np.cos(theta) * (u_f + u_r)
        Jx[4,2] = - (1 / self.m) * np.sin(theta) * (u_f + u_r)

        # Input Jacobian
        Ju = np.zeros((self.n_states, self.n_inputs))
        Ju[3,:] = - (1 / self.m) * np.sin(theta)
        Ju[4,:] = (1 / self.m) * np.cos(theta)
        Ju[5,0] = self.l / self.I
        Ju[5,1] =  - self.l / self.I
        return Jx, Ju
