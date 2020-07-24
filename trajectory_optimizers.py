import controlpy
import control_utils
import controllers
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
LIN_TRAJ = "linearization state trajectory"
LIN_INPUT_TRAJ = "linearization input trajectory"

class Traj_Opt(ABC):

	@abstractmethod
	def __init__(self, dt):
		self.dt = dt

	@abstractmethod
	def build(self):
		raise(NotImplementedError)

	@abstractmethod
	def solve_iteration(self):
		raise(NotImplementedError)

class SCP_Traj_Opt(Traj_Opt):

	def __init__(self, N, Q, R, state_reference, input_reference, state_constraints, input_constraints, tolerance=1e-3, regularization=1e2, solver="OSQP"):
		super(SCP_Traj_Opt, self).__init__(np.NaN)
		self.Q = Q
		self.R = R
		self.P = Q
		self.N = N

		self.A = None
		self.B = None
		self.C = None

		self.state_reference = state_reference
		self.input_reference = input_reference

		self.state_constraints = state_constraints
		self.input_constraints = input_constraints
		self.terminal_constraint = state_constraints
		
		self.dx = None
		self.du = None
		self.slack = None
		self.terminal_slack = None
		self.problem = None
		self.problem_parameters = None
		self.cost = None
		self.feasible = False
		self.converged = False

		self.tolerance = tolerance
		self.regularization = regularization
		self.traj_list = []
		self.backward_traj_list = []
		self.input_traj_list = []
		self.backward_input_traj_list = []
		self.slack_traj_list = []
		self.terminal_slack_list = []
		self.solution_costs = []
		self.i = 0

		self.solver = cp.OSQP if solver is "OSQP" else cp.ECOS

	def build_solver(self):
		(x_param, u_param, dx, du,
			slack, terminal_slack, A, B, 
			state_reference, input_reference, 
			state_constraints, input_constraints, 
			terminal_constraint, problem) =  mpc_solvers.SCP_ftocp_solver(self.N, 
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.terminal_constraint[0].shape[0], 
													self.Q, self.R, self.P,
													regularization=self.regularization)

		self.dx = dx
		self.du = du
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.problem = problem
		self.problem_parameters = {LIN_TRAJ:x_param, LIN_INPUT_TRAJ:u_param, MODEL_A:A, MODEL_B:B, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints, 
									TERMINAL_CONSTRAINT:terminal_constraint}
		self.cost = None
		self.feasible = True

	def build(self):
		self.build_solver()
		self.set_basic_parameters()
		self.problem_parameters[TERMINAL_CONSTRAINT][0].value = self.terminal_constraint[0]
		self.problem_parameters[TERMINAL_CONSTRAINT][1].value = self.terminal_constraint[1]

	def set_basic_parameters(self):
		for i in range(self.N):
			self.problem_parameters[STATE_REFERENCE][i].value = self.state_reference
			self.problem_parameters[INPUT_REFERENCE][i].value = self.input_reference
			self.problem_parameters[STATE_CONSTRAINTS][i][0].value = self.state_constraints[0]
			self.problem_parameters[STATE_CONSTRAINTS][i][1].value = self.state_constraints[1]
			self.problem_parameters[INPUT_CONSTRAINTS][i][0].value = self.input_constraints[0]
			self.problem_parameters[INPUT_CONSTRAINTS][i][1].value = self.input_constraints[1]

		self.problem_parameters[STATE_REFERENCE][self.N].value = self.state_reference

	def solve_ftocp(self):
		"""
		Solves the finite time optimal control problem for mpc
		returns predicted state and input trajectories, cost function, and feasibility of the problem
		"""
		self.problem.solve(solver=self.solver)
		
		if self.problem.status is not "infeasible": 
			self.cost = self.problem.value
			self.feasible = True
			return self.dx.value, self.du.value
		else: 
			self.feasible = False
			return None, None

	def solve_iteration(self, forward_x_traj, forward_u_traj, A, B):
		self.traj_list.append(forward_x_traj)
		self.input_traj_list.append(forward_u_traj)
		for i in range(self.N):
			self.problem_parameters[MODEL_A][i].value = A[i]
			self.problem_parameters[MODEL_B][i].value = B[i]
		self.problem_parameters[LIN_TRAJ].value = forward_x_traj
		self.problem_parameters[LIN_INPUT_TRAJ].value  = forward_u_traj

		dx, du = self.solve_ftocp()

		if self.feasible:
			x_traj = self.traj_list[-1] + dx
			u_traj = self.input_traj_list[-1] + du
			self.i += 1
			self.backward_traj_list.append(x_traj)
			self.backward_input_traj_list.append(u_traj)
			self.slack_traj_list.append(self.slack.value)
			self.terminal_slack_list.append(self.terminal_slack.value)
			self.solution_costs.append(self.cost)
			self.converged = np.all(np.linalg.norm(dx, axis=0) <= self.tolerance) and np.all(np.linalg.norm(du, axis=0) <= self.tolerance)  

		return x_traj, u_traj, self.converged

	def get_controller(self):
		controller = controllers.Open_Loop_Controller()
		controller.build(self.backward_input_traj_list[-1])
		return controller

class SCP_LMPC_Traj_Opt(SCP_Traj_Opt):

	def __init__(self, N, Q, R, state_reference, input_reference, state_constraints, input_constraints, n_safe_set, tolerance=1e-3, regularization=1e2, solver="OSQP"):
		super(SCP_LMPC_Traj_Opt, self).__init__(N, Q, R, state_reference, input_reference, state_constraints, input_constraints,
																 tolerance=tolerance, regularization=regularization, solver=solver)
		self.n_safe_set = n_safe_set
		self.P = None
		self.safe_set = None
		self.value_function = None

	def build_solver(self):
		(x_param, u_param, dx, du,
			multipliers, slack, terminal_slack, 
			safe_set, value_function, A, B, 
			state_reference, input_reference, 
			state_constraints, input_constraints, problem) = mpc_solvers.SCP_LMPC_ftocp_solver(self.N, self.n_safe_set, 
													self.state_constraints[0].shape[0], 
													self.input_constraints[0].shape[0], 
													self.Q, self.R,
													regularization=self.regularization)

		self.dx = dx
		self.du = du
		self.multipliers = multipliers
		self.safe_set = safe_set
		self.value_function = value_function
		self.slack = slack
		self.terminal_slack = terminal_slack
		self.problem = problem
		self.problem_parameters = {LIN_TRAJ:x_param, LIN_INPUT_TRAJ:u_param, MODEL_A:A, MODEL_B:B, 
									STATE_REFERENCE:state_reference, INPUT_REFERENCE:input_reference, 
									STATE_CONSTRAINTS:state_constraints, INPUT_CONSTRAINTS:input_constraints}
		self.cost = None
		self.feasible = True

	def build(self):
		self.build_solver()
		self.set_basic_parameters()

	def set_safe_set(self, safe_set, value_function):
		self.safe_set.value = safe_set
		self.value_function.value = value_function

	def reset(self):
		self.i = 0
		self.traj_list = []
		self.backward_traj_list = []
		self.input_traj_list = []
		self.backward_input_traj_list = []
		self.slack_traj_list = []
		self.terminal_slack_list = []
		self.solution_costs = []

