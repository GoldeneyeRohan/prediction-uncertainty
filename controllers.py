import controlpy
import control_utils
import cvxpy as cp
import mpc_solvers
import numpy as np
import polytope
import scipy
from abc import ABC, abstractmethod
import time


MODEL_A = "A"
MODEL_B = "B"
MODEL_C = "C"
A_HAT = "A_hat"
B_HAT = "B_hat"
C_HAT = "C_hat"
STATE_REFERENCE = "state reference"
INPUT_REFERENCE = "input reference"
STATE_CONSTRAINTS = "state constraints"
INPUT_CONSTRAINTS = "input constraints"
TERMINAL_CONSTRAINT = "terminal constraint"
INIT_CONSTRAINT = "init constraint"

class Controller(ABC):

	@abstractmethod
	def __init__(self, dt):
		self.dt = dt

	@abstractmethod
	def build(self):
		raise(NotImplementedError)

	@abstractmethod
	def solve(self):
		raise(NotImplementedError)

class PID(Controller):

	def __init__(self, P, I, D, state_reference, input_reference, dt, input_constraints):
		super(PID, self).__init__(dt)
		self.Kp = P
		self.Ki = I
		self.Kd = D
		self.state_reference = state_reference
		self.input_reference = input_reference

		self.input_constraints = input_constraints

		self.e  = None
		self.de = None
		self.ei = None

	def build(self):
		self.e  = 0
		self.de = 0
		self.ei = 0

	def solve(self, x):
		e = self.state_reference - x
		de = (e - self.e) / self.dt
		ei = self.ei + e * self.dt

		u = self.Kp * e + self.Kd * de + self.Ki * ei + self.input_reference

		self.e = e
		self.de = de

		if u < self.input_constraints[0]:
			return self.input_constraints[0]
		elif u > self.input_constraints[1]:
			return self.input_constraints[1]
		else:
			self.ei = ei
			return u

	def set_reference(self, state_reference, input_reference):
		self.state_reference = state_reference
		self.input_reference = input_reference

class LTI_MPC_Controller(Controller):
	""" 
	LTI MPC Controller
	- keeps constant state reference
	- constant dynamics
	- constant state and input constraints
	- constant terminal constraint
	"""

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints):
		super(LTI_MPC_Controller, self).__init__(np.NaN)

		self.Q = Q
		self.R = R
		self.P = None
		self.N = N

		self.A = A
		self.B = B
		self.C = C

		self.state_reference = state_reference
		self.input_reference = np.zeros(R.shape[0])

		self.state_constraints = state_constraints
		self.input_constraints = input_constraints
		self.terminal_constraint = None
		
		self.x_traj = None
		self.u_traj = None
		self.slack = None
		self.terminal_slack = None
		self.problem = None
		self.problem_parameters = None
		self.x0 = None
		self.cost = None
		self.feasible = False

	@abstractmethod
	def build(self):
		raise(NotImplementedError)

	def build_solver(self):
		(x0, x_traj, u_traj, slack, terminal_slack,
			A, B, C, 
			state_reference, input_reference, 
			state_constraints, input_constraints, 
			terminal_constraint, problem) =  mpc_solvers.LTV_ftocp_solver(self.N, 
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.terminal_constraint[0].shape[0], self.Q, self.R, self.P)

		self.x_traj = x_traj
		self.u_traj = u_traj
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.problem = problem
		self.problem_parameters = {MODEL_A:A, MODEL_B:B, MODEL_C:C, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints, 
									TERMINAL_CONSTRAINT:terminal_constraint}
		self.x0 = x0
		self.cost = None
		self.feasible = True

	def set_basic_parameters(self):
		for i in range(self.N):
			self.problem_parameters[MODEL_A][i].value = self.A
			self.problem_parameters[MODEL_B][i].value = self.B
			self.problem_parameters[MODEL_C][i].value = self.C
			self.problem_parameters[STATE_REFERENCE][i].value = self.state_reference
			self.problem_parameters[INPUT_REFERENCE][i].value = self.input_reference
			self.problem_parameters[STATE_CONSTRAINTS][i][0].value = self.state_constraints[0]
			self.problem_parameters[STATE_CONSTRAINTS][i][1].value = self.state_constraints[1]
			self.problem_parameters[INPUT_CONSTRAINTS][i][0].value = self.input_constraints[0]
			self.problem_parameters[INPUT_CONSTRAINTS][i][1].value = self.input_constraints[1]

		self.problem_parameters[STATE_REFERENCE][self.N].value = self.state_reference

	def solve_ftocp(self, x0):
		"""
		Solves the finite time optimal control problem for mpc
		returns predicted state and input trajectories, cost function, and feasibility of the problem
		"""
		self.x0.value = x0
		# t = time.time()
		self.problem.solve(solver=cp.ECOS)
		# tf = time.time()
		# print(1 / (tf - t))
		if self.problem.status is not "infeasible": 
			self.cost = self.problem.value
			self.feasible = True
			return self.u_traj.value[:,0]
		else: 
			self.feasible = False
			return None

class LTI_MPC_Tracker(LTI_MPC_Controller):
	"""
	"""

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints):
		super(LTI_MPC_Tracker, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints)
		self.P = Q
		self.terminal_constraint = self.state_constraints

	def build(self):
		self.build_solver()
		self.set_basic_parameters()
		self.problem_parameters[TERMINAL_CONSTRAINT][0].value = self.terminal_constraint[0]
		self.problem_parameters[TERMINAL_CONSTRAINT][1].value = self.terminal_constraint[1]

	def solve(self, x0):
		return self.solve_ftocp(x0)

class LTI_MPC(LTI_MPC_Controller):
	"""
	"""

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints):
		super(LTI_MPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints)

	def build(self):
		K, P, _ = controlpy.synthesis.controller_lqr_discrete_time(self.A, self.B, self.Q, self.R)
		self.P = P
		X = polytope.Polytope(self.state_constraints[0], self.state_constraints[1])
		U = polytope.Polytope(self.input_constraints[0], self.input_constraints[1])
		X_terminal = control_utils.maximal_invariant(X, self.A, B=self.B, K=K, U=U)
		self.terminal_constraint = (X_terminal.A, X_terminal.b)

		self.build_solver()
		self.set_basic_parameters()
		self.problem_parameters[TERMINAL_CONSTRAINT][0].value = self.terminal_constraint[0]
		self.problem_parameters[TERMINAL_CONSTRAINT][1].value = self.terminal_constraint[1]

	def solve(self, x0):
		return self.solve_ftocp(x0)

class LTI_Tube_MPC(LTI_MPC_Controller):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints):
		super(LTI_Tube_MPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints)
		self.K = None
		self.init_constraint = None

	def build_solver(self):
		(x0, x_traj, u_traj, slack, terminal_slack, 
			A, B, C, 
			state_reference, input_reference, 
			state_constraints, input_constraints, 
			init_constraint, terminal_constraint, 
			problem) =  mpc_solvers.LTV_tube_ftocp_solver(self.N, 
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.init_constraint[0].shape[0],
													self.terminal_constraint[0].shape[0],  
													self.Q, self.R, self.P)

		self.x_traj = x_traj
		self.u_traj = u_traj
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.problem = problem
		self.problem_parameters = {MODEL_A:A, MODEL_B:B, MODEL_C:C, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints, 
									INIT_CONSTRAINT:init_constraint, TERMINAL_CONSTRAINT:terminal_constraint}
		self.x0 = x0
		self.cost = None
		self.feasible = True

	def build(self, w):
		K, P, _ = controlpy.synthesis.controller_lqr_discrete_time(self.A, self.B, self.Q, self.R)
		self.P = P
		self.K = K

		W = polytope.Polytope(w[0], w[1])
		M = control_utils.minimal_invariant(self.A - self.B @ K, W)

		X = polytope.Polytope(self.state_constraints[0], self.state_constraints[1])
		U = polytope.Polytope(self.input_constraints[0], self.input_constraints[1])
		
		X_terminal, _ = control_utils.robust_maximal_invariant(X, W, M, self.A, B=self.B, K=K, U=U)
		X_bar = control_utils.pontryagin_difference(X, M)
		KM = control_utils.poly_transform(M, K)
		U_bar = control_utils.pontryagin_difference(U, KM)

		self.init_constraint = (M.A, M.b)
		self.terminal_constraint = (X_terminal.A, X_terminal.b)
		self.state_constraints = (X_bar.A, X_bar.b)
		self.input_constraints = (U_bar.A, U_bar.b)

		self.build_solver()
		self.set_basic_parameters()
		self.problem_parameters[INIT_CONSTRAINT][0].value = self.init_constraint[0]
		self.problem_parameters[INIT_CONSTRAINT][1].value = self.init_constraint[1]
		self.problem_parameters[TERMINAL_CONSTRAINT][0].value = self.terminal_constraint[0]
		self.problem_parameters[TERMINAL_CONSTRAINT][1].value = self.terminal_constraint[1]

	def solve(self, x0):
		u_nominal = self.solve_ftocp(x0)
		if u_nominal is not None:
			x_0 = self.x_traj[:,0].value
			u =  - self.K @ (x0 - x_0) + u_nominal
			return u
		else:
			return None

class LTI_LMPC(LTI_MPC_Controller):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=None):
		super(LTI_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints)
		self.ss_size_fixed = n_safe_set is not None
		self.n_safe_set = n_safe_set
		self.safe_set = None
		self.value_function = None
		self.multipliers = None
		self.traj_list = []
		self.input_traj_list = []
		self.value_func_list = []
		self.i = 0

	def build_solver(self):
		(x0, x_traj, u_traj, multipliers, slack, terminal_slack,
			safe_set, value_function, 
			A, B, C, 
			state_reference, input_reference, 
			state_constraints, input_constraints, 
			problem) =  mpc_solvers.LTV_LMPC_ftocp_solver(self.N, self.n_safe_set,
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.Q, self.R)

		self.x_traj = x_traj
		self.u_traj = u_traj
		self.multipliers = multipliers
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.safe_set = safe_set
		self.value_function = value_function
		self.problem = problem
		self.problem_parameters = {MODEL_A:A, MODEL_B:B, MODEL_C:C, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints}
		self.x0 = x0
		self.cost = None
		self.feasible = True

	def build(self):
		if not self.ss_size_fixed:
			self.n_safe_set = np.sum([l.shape[1] for l in self.traj_list])
		self.build_solver()
		self.set_basic_parameters()

	def add_trajectory(self, state_traj, input_traj, value_function):
		self.traj_list.append(state_traj)
		self.input_traj_list.append(input_traj)
		self.value_func_list.append(value_function)
		self.i = 0
		if not self.ss_size_fixed:
			self.build()
			self.safe_set.value = np.hstack(self.traj_list)
			self.value_function.value = np.hstack(self.value_func_list)

	def solve(self, x0):
		self.i += 1
		return self.solve_ftocp(x0)

class LTI_Tube_LMPC(LTI_LMPC):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, disturbance_set, n_safe_set=None, minimal_n=15):
		super(LTI_Tube_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=n_safe_set)
		K, P, _ = controlpy.synthesis.controller_lqr_discrete_time(self.A, self.B, self.Q, self.R)
		self.K = K
		self.init_constraint = None
		self.disturbance_set = disturbance_set
		self.minimal_n = minimal_n


	def build_solver(self):
		(x0, init_constraint, 
			x_traj, u_traj, multipliers, slack, terminal_slack,
			safe_set, value_function, 
			A, B, C, 
			state_reference, input_reference, 
			state_constraints, input_constraints, 
			problem) =  mpc_solvers.LTV_tube_LMPC_ftocp_solver(self.N, self.n_safe_set,
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.init_constraint[0].shape[0], 
													self.Q, self.R)

		self.x_traj = x_traj
		self.u_traj = u_traj
		self.multipliers = multipliers
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.safe_set = safe_set
		self.value_function = value_function
		self.problem = problem
		self.problem_parameters = {MODEL_A:A, MODEL_B:B, MODEL_C:C, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints,
									INIT_CONSTRAINT:init_constraint}
		self.x0 = x0
		self.cost = None
		self.feasible = True

	def build(self):
		if not self.ss_size_fixed:
			self.n_safe_set = np.sum([l.shape[1] for l in self.traj_list])

		if self.init_constraint is None:
			W = polytope.Polytope(self.disturbance_set[0], self.disturbance_set[1])
			M = control_utils.minimal_invariant(self.A - self.B @ self.K, W, n=self.minimal_n)

			X = polytope.Polytope(self.state_constraints[0], self.state_constraints[1])
			U = polytope.Polytope(self.input_constraints[0], self.input_constraints[1])
		
			X_bar = control_utils.pontryagin_difference(X, M)
			KM = control_utils.poly_transform(M, self.K)
			U_bar = control_utils.pontryagin_difference(U, KM)

			self.init_constraint = (M.A, M.b)
			self.state_constraints = (X_bar.A, X_bar.b)
			self.input_constraints = (U_bar.A, U_bar.b)

		self.build_solver()
		self.set_basic_parameters()
		self.problem_parameters[INIT_CONSTRAINT][0].value = self.init_constraint[0]
		self.problem_parameters[INIT_CONSTRAINT][1].value = self.init_constraint[1]


	def solve(self, x0):
		u_nominal = self.solve_ftocp(x0)
		self.i += 1
		if u_nominal is not None:
			x_0 = self.x_traj[:,0].value
			u =  - self.K @ (x0 - x_0) + u_nominal
			return u
		else:
			return None

class LTI_Robust_LMPC(LTI_LMPC):
	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, disturbance_set, n_safe_set=None, minimal_n=15):
		super(LTI_Robust_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set)
		K, P, _ = controlpy.synthesis.controller_lqr_discrete_time(self.A, self.B, self.Q, self.R)
		self.K = K
		self.disturbance_set = disturbance_set
		self.minimal_n = minimal_n
		self.minimal_invariant = None

	def build(self):
		if not self.ss_size_fixed:
			self.n_safe_set = np.sum([l.shape[1] for l in self.traj_list])

		if self.minimal_invariant is None:
			W = polytope.Polytope(self.disturbance_set[0], self.disturbance_set[1])
			M = control_utils.minimal_invariant(self.A - self.B @ self.K, W, n=self.minimal_n)

			X = polytope.Polytope(self.state_constraints[0], self.state_constraints[1])
			U = polytope.Polytope(self.input_constraints[0], self.input_constraints[1])
		
			X_bar = control_utils.pontryagin_difference(X, M)
			KM = control_utils.poly_transform(M, self.K)
			U_bar = control_utils.pontryagin_difference(U, KM)

			self.state_constraints = (X_bar.A, X_bar.b)
			self.input_constraints = (U_bar.A, U_bar.b)
			self.minimal_invariant = (M.A, M.b)

		self.build_solver()
		self.set_basic_parameters()

	def solve(self, x0):
		if self.i == 0:
			x_0 = x0
			u_nominal = self.solve_ftocp(x0)
		else: 
			x_0 = self.x_traj[:,1].value
			u_nominal = self.solve_ftocp(x_0)

		if u_nominal is not None:
			u = u_nominal - self.K @ (x0 - x_0)
			self.i += 1
			return u 
		else: 
			return None

class LTI_Stochastic_LMPC(LTI_LMPC):
	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, disturbance_covariance, n_safe_set=None):
		super(LTI_Stochastic_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set)
		K, P, _ = controlpy.synthesis.controller_lqr_discrete_time(self.A, self.B, self.Q, self.R)
		self.K = K
		self.disturbance_covariance = disturbance_covariance
		self.minimal_invariant_covariance = None

	def build(self):
		if not self.ss_size_fixed:
			self.n_safe_set = np.sum([l.shape[1] for l in self.traj_list])

		if self.minimal_invariant_covariance is None:
			Sigma_inf = scipy.linalg.solve_discrete_lyapunov(self.A - self.B @ self.K, self.disturbance_covariance)

			X = polytope.Polytope(self.state_constraints[0], self.state_constraints[1])
			U = polytope.Polytope(self.input_constraints[0], self.input_constraints[1])
		
			X_bar = control_utils.pontryagin_difference(X, np.linalg.inv(Sigma_inf))
			U_bar = control_utils.pontryagin_difference(U, np.linalg.inv(Sigma_inf), K=self.K)

			self.state_constraints = (X_bar.A, X_bar.b)
			self.input_constraints = (U_bar.A, U_bar.b)
			self.minimal_invariant_covariance = Sigma_inf

		self.build_solver()
		self.set_basic_parameters()

	def solve(self, x0):
		if self.i == 0:
			x_0 = x0
			u_nominal = self.solve_ftocp(x0)
		else: 
			x_0 = self.x_traj[:,1].value
			u_nominal = self.solve_ftocp(x_0)

		if u_nominal is not None:
			u = u_nominal - self.K @ (x0 - x_0)
			self.i += 1
			return u 
		else: 
			return None

class LTV_LMPC(LTI_LMPC):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=None):
		super(LTV_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=n_safe_set)
		self.A = [A] * self.N 
		self.B = [B] * self.N 
		self.C = [C] * self.N

	def set_basic_parameters(self):
		for i in range(self.N):
			self.problem_parameters[MODEL_A][i].value = self.A[i]
			self.problem_parameters[MODEL_B][i].value = self.B[i]
			self.problem_parameters[MODEL_C][i].value = self.C[i]
			self.problem_parameters[STATE_REFERENCE][i].value = self.state_reference
			self.problem_parameters[INPUT_REFERENCE][i].value = self.input_reference
			self.problem_parameters[STATE_CONSTRAINTS][i][0].value = self.state_constraints[0]
			self.problem_parameters[STATE_CONSTRAINTS][i][1].value = self.state_constraints[1]
			self.problem_parameters[INPUT_CONSTRAINTS][i][0].value = self.input_constraints[0]
			self.problem_parameters[INPUT_CONSTRAINTS][i][1].value = self.input_constraints[1]

		self.problem_parameters[STATE_REFERENCE][self.N].value = self.state_reference

	def set_models(self, A, B, C):
		self.A = A
		self.B = B
		self.C = C
		self.set_basic_parameters()

class LBLMPC(LTI_Tube_LMPC):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, disturbance_set, n_safe_set=None, minimal_n=15):
		super(LBLMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, disturbance_set, n_safe_set=None, minimal_n=minimal_n)
		self.A_hat = [A] * self.N
		self.B_hat = [B] * self.N
		self.C_hat = [C] * self.N
		self.x_hat_traj = None
		self.multipliers_hat = None
		self.terminal_slack_hat = None

	def build_solver(self):
		(x0, init_constraint, x_traj, u_traj, multipliers, slack, 
				x_hat_traj, multipliers_hat,
				terminal_slack, terminal_slack_hat, 
				safe_set, value_function, 
				A, B, C, A_hat, B_hat, C_hat, 
				state_reference, input_reference, 
				state_constraints, input_constraints, problem) =  mpc_solvers.LBLMPC_ftocp_solver(self.N, self.n_safe_set,
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.init_constraint[0].shape[0],
													self.Q, self.R)

		self.x_traj = x_traj
		self.u_traj = u_traj
		self.multipliers = multipliers
		self.x_hat_traj = x_hat_traj
		self.multipliers_hat = multipliers_hat
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.terminal_slack_hat = terminal_slack_hat
		self.safe_set = safe_set
		self.value_function = value_function
		self.problem = problem
		self.problem_parameters = {MODEL_A:A, MODEL_B:B, MODEL_C:C, A_HAT:A_hat, B_HAT:B_hat, C_HAT:C_hat,
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints, INIT_CONSTRAINT:init_constraint}
		self.x0 = x0
		self.cost = None
		self.feasible = True

	def build(self): 
		super().build()
		for i in range(self.N):
			self.problem_parameters[A_HAT][i].value = self.A_hat[i]
			self.problem_parameters[B_HAT][i].value = self.B_hat[i]
			self.problem_parameters[C_HAT][i].value = self.C_hat[i]

	def set_models(self, A_hat, B_hat, C_hat):
		self.A_hat = A_hat
		self.B_hat = B_hat
		self.C_hat = C_hat
		for i in range(self.N):
			self.problem_parameters[A_HAT][i].value = self.A_hat[i]
			self.problem_parameters[B_HAT][i].value = self.B_hat[i]
			self.problem_parameters[C_HAT][i].value = self.C_hat[i]

class Local_LTV_LMPC(LTV_LMPC):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set_it, n_safe_set):
		super(Local_LTV_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=n_safe_set)
		self.n_safe_set_it = n_safe_set_it
		self.x_ss = None

	def select_safe_set(self, x):
		n_trajectories = len(self.traj_list)
		points_per_trajectory = control_utils.split_n(self.n_safe_set, min(n_trajectories, self.n_safe_set_it))
		best_iteration_order = np.argsort([q[0] for q in self.value_func_list])
		(safe_set, input_safe_set, value_function, successor_safe_set) = zip(
		 			*[control_utils.select_points(x, n, self.traj_list[traj_index], self.input_traj_list[traj_index], self.value_func_list[traj_index]) 
		 					for n, traj_index in zip(points_per_trajectory, best_iteration_order)])

		safe_set = np.hstack(safe_set)
		input_safe_set = np.hstack(input_safe_set)
		value_function = np.hstack(value_function)
		successor_safe_set = np.hstack(successor_safe_set)

		return safe_set, input_safe_set, value_function, successor_safe_set

	def solve(self, x0):
		# import pdb; pdb.set_trace()
		x_ss = self.x_ss if (self.x_ss is not None and self.i != 0) else x0
		self.safe_set.value, _, self.value_function.value, successor_safe_set = self.select_safe_set(x_ss)
		u = self.solve_ftocp(x0)
		if self.feasible:
			self.x_ss = successor_safe_set @ self.multipliers.value 
		self.i += 1
		return u

class True_LTV_LMPC(LTV_LMPC):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints):
		super(True_LTV_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=0)
		self.min_ss_traj_length = 1e9

	def add_trajectory(self, state_traj, input_traj, value_function):
		self.traj_list.append(state_traj)
		self.input_traj_list.append(input_traj)
		self.value_func_list.append(value_function)
		self.min_ss_traj_length = min(self.min_ss_traj_length, state_traj.shape[1])
		self.n_safe_set += 1
		self.i = 0
		self.build()

	def select_safe_set(self, i):
		i_ss = min(i + self.N, self.min_ss_traj_length - 1)
		safe_set = [traj[:,i_ss] for traj in self.traj_list]
		value_function = [q[i_ss] for q in self.value_func_list]
		return np.vstack(safe_set).T, np.array(value_function)

	def solve(self, x0):
		self.safe_set.value, self.value_function.value = self.select_safe_set(self.i)
		self.i += 1
		return self.solve_ftocp(x0)

class LTV_Tube_LMPC(LTV_LMPC):

	def __init__(self, A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, minimal_invariant, n_safe_set=None):
		super(LTV_Tube_LMPC, self).__init__(A, B, C, N, Q, R, state_reference, state_constraints, input_constraints, n_safe_set=n_safe_set)
		K, P, _ = controlpy.synthesis.controller_lqr_discrete_time(self.A[0], self.B[0], self.Q, self.R)
		self.K = K
		self.minimal_invariant = minimal_invariant
		self.init_constraint = None

	def build_solver(self):
		LTI_Tube_LMPC.build_solver(self)

	def build(self):
		if not self.ss_size_fixed:
			self.n_safe_set = np.sum([l.shape[1] for l in self.traj_list])

		if self.init_constraint is None:
			M = polytope.Polytope(*self.minimal_invariant)
			X = polytope.Polytope(self.state_constraints[0], self.state_constraints[1])
			U = polytope.Polytope(self.input_constraints[0], self.input_constraints[1])
		
			X_bar = control_utils.pontryagin_difference(X, M)
			KM = control_utils.poly_transform(M, self.K)
			U_bar = control_utils.pontryagin_difference(U, KM)

			self.init_constraint = (M.A, M.b)
			self.state_constraints = (X_bar.A, X_bar.b)
			self.input_constraints = (U_bar.A, U_bar.b)

		self.build_solver()
		self.set_basic_parameters()
		self.problem_parameters[INIT_CONSTRAINT][0].value = self.init_constraint[0]
		self.problem_parameters[INIT_CONSTRAINT][1].value = self.init_constraint[1]

	def solve(self, x0):
		return LTI_Tube_LMPC.solve(self, x0)

class SCP_Traj_Opt(Controller):

	def __init__(self, N, Q, R, state_reference, state_constraints, input_constraints, tolerance=1e-3, regularization=1e-2):
		super(SCP_Traj_Opt, self).__init__(np.NaN)
		self.Q = Q
		self.R = R
		self.P = Q
		self.N = N

		self.A = None
		self.B = None
		self.C = None

		self.state_reference = state_reference
		self.input_reference = np.zeros(R.shape[0])

		self.state_constraints = state_constraints
		self.input_constraints = input_constraints
		self.terminal_constraint = state_constraints
		
		self.x_traj = None
		self.u_traj = None
		self.slack = None
		self.terminal_slack = None
		self.problem = None
		self.problem_parameters = None
		self.x0 = None
		self.cost = None
		self.feasible = False
		self.converged = False

		self.solution_costs
		self.tolerance = tolerance
		self.regularization = regularization
		self.traj_list = []
		self.input_traj_list = []
		self.slack_traj_list = []
		self.terminal_slack_list = []
		self.i = 0

	def build_solver(self):
		(x0, x_traj, u_traj, slack, terminal_slack,
			A, B, C, 
			state_reference, input_reference, 
			state_constraints, input_constraints, 
			terminal_constraint, problem) =  mpc_solvers.SCP_ftocp_solver(self.N, 
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.terminal_constraint[0].shape[0], 
													self.Q, self.R, self.P,
													regularization=self.regularization)

		self.x_traj = x_traj
		self.u_traj = u_traj
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.problem = problem
		self.problem_parameters = {MODEL_A:A, MODEL_B:B, MODEL_C:C, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints, 
									TERMINAL_CONSTRAINT:terminal_constraint}
		self.x0 = x0
		self.cost = None
		self.feasible = True

	def build(self, x_init_traj, u_init_traj):
		self.traj_list.append(x_init_traj)
		self.input_traj_list.append(u_init_traj)
		self.slack_traj_list.append(None)
		self.terminal_slack_list.append(None)
		self.build_solver()
		self.set_basic_parameters()

	def set_basic_parameters(self):
		for i in range(self.N):
			self.problem_parameters[STATE_REFERENCE][i].value = self.state_reference
			self.problem_parameters[INPUT_REFERENCE][i].value = self.input_reference
			self.problem_parameters[STATE_CONSTRAINTS][i][0].value = self.state_constraints[0]
			self.problem_parameters[STATE_CONSTRAINTS][i][1].value = self.state_constraints[1]
			self.problem_parameters[INPUT_CONSTRAINTS][i][0].value = self.input_constraints[0]
			self.problem_parameters[INPUT_CONSTRAINTS][i][1].value = self.input_constraints[1]

		self.problem_parameters[STATE_REFERENCE][self.N].value = self.state_reference
		self.problem_parameters[TERMINAL_CONSTRAINT][0].value = self.terminal_constraint[0]
		self.problem_parameters[TERMINAL_CONSTRAINT][1].value = self.terminal_constraint[1]

	def solve_ftocp(self):
		"""
		Solves the finite time optimal control problem for mpc
		returns predicted state and input trajectories, cost function, and feasibility of the problem
		"""
		self.problem.solve(solver=cp.ECOS)
		
		if self.problem.status is not "infeasible": 
			self.cost = self.problem.value
			self.feasible = True
			return self.x_traj.value, self.u_traj.value
		else: 
			self.feasible = False
			return None, None

	def solve_iteration(A, B):
		for i in range(self.N):
			self.problem_parameters[MODEL_A][i].value = A[i]
			self.problem_parameters[MODEL_B][i].value = B[i]
			self.problem_parameters[MODEL_C][i].value = self.x_traj_list[-1][:,i+1]

		x_traj, u_traj = self.solve_ftocp()

		if self.feasible:
			self.i += 1
			self.converged = np.all(np.linalg.norm(x_traj, axis=0) <= self.tolerance) and np.all(np.linalg.norm(u_traj, axis=0) <= self.tolerance)  

		return x_traj, u_traj, self.converged





