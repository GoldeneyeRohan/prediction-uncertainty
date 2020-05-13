import numpy as np 
import cvxpy as cp

def get_dynamics_constraints(x, u, A, B, C, N):
	dynamics_constraints = []
	for i in range(N):
		dynamics_constraints += [x[:, i+1] == A[i] @ x[:,i] + B[i] @ u[:,i] + C[i]]
	return dynamics_constraints
	
def get_state_constraints(x, state_limits, N):
	state_constraints = []
	for i in range(N):
		state_constraints += [state_limits[i][0] @ x[:,i] <= state_limits[i][1]]
	return state_constraints

def get_input_constraints(u, input_limits, N):
	input_constraints = []
	for i in range(N):
		input_constraints += [input_limits[i][0] @ u[:,i] <= input_limits[i][1]]	
	return input_constraints

def get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N):
	trajectory_cost = 0
	for i in range(N):
		trajectory_cost += cp.quad_form((u[:,i] - input_reference[i]), R)
		trajectory_cost += cp.quad_form((x[:,i] - state_reference[i]), Q)
	return trajectory_cost
	
def get_dynamics_parameters(n_states, n_inputs, N):
	A = [cp.Parameter((n_states, n_states)) for _ in range(N)]
	B = [cp.Parameter((n_states, n_inputs)) for _ in range(N)]
	C = [cp.Parameter(n_states) for _ in range(N)]
	return A, B, C

def get_reference_parameters(n_states, n_inputs, N):
	input_reference = [cp.Parameter(n_inputs) for _ in range(N)]
	state_reference = [cp.Parameter(n_states) for _ in range(N + 1)]
	return state_reference, input_reference

def get_constraint_parameters(n_states, n_inputs, m_state_constraints, m_input_constraints, N):
	state_limits = [(cp.Parameter((m_state_constraints, n_states)), cp.Parameter(m_state_constraints)) for _ in range(N)]
	input_limits = [(cp.Parameter((m_input_constraints, n_inputs)), cp.Parameter(m_input_constraints)) for _ in range(N)]
	return state_limits, input_limits

def LTV_ftocp_solver(N, m_state_constraints, m_input_constraints, m_terminal_set, Q, R, P):
	"""
	Solves the Finite Time Optimal Control Problem for a tracking MPC policy.
	- A = (List of N) state dynamics matrix (nxn matrix)
	- B = (List of N) input dynamics matrix (nxm matrix)
	- C = (List of N) affine dynamics terms (nx1 vector)
	- N = prediction horizon
	- state_limits = (List of N) tuple (H,h) defining the set X_i = {x | Hx <= h} 
	- input_constraints = (List of N) tuple (S,s) defining the set U_i = {u | Su<=s}
	- terminal_set = tuple (G,g) defining the terminal constraint set X_N+1 = {x | Gx <= g} 
	- Q = state dependant stage cost (x^TQx) (nxn matrix)
	- R = input dependant stage cost (u^TRu) (mxm matrix)
	- state reference = (List of N + 1) state reference vectors
	- input reference = (List of N) input reference vectors
	Problem solved:
		min \sum_{i=1}^{N} x_i^TQx_i + u_i^TRu_i + x_{N+1}^T P x_{N + 1} : s.t. x_0 = x(0), dynamics, state + input constraints + terminal constraint 
	returns x0 parameter, state_trajectory variable, input_trajectory variable, problem
	"""
	# Setup
	n = Q.shape[0]
	m = R.shape[0]

	# Decision Variables
	x = cp.Variable((n, N + 1))
	u = cp.Variable((m, N))

	# Problem Parameters
	x0_param = cp.Parameter(n)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)
	terminal_set = (cp.Parameter((m_terminal_set, n)), cp.Parameter(m_terminal_set))

	# Constraints
	init_constraint = [x[:,0] == x0_param]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraint = [terminal_set[0] @ x[:, N] <= terminal_set[1]]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += cp.quad_form((x[:,N] - state_reference[N]), P)
	cost = cp.Minimize(trajectory_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraint 
	problem = cp.Problem(cost, constraints)

	return x0_param, x, u, A, B, C, state_reference, input_reference, state_limits, input_limits, terminal_set, problem

def LTV_tube_ftocp_solver(N, m_state_constraints, m_input_constraints, m_init_set, m_terminal_set, Q, R, P):
	n = Q.shape[0]
	m = R.shape[0]

	# Decision Variables
	x = cp.Variable((n, N + 1))
	u = cp.Variable((m, N))

	# Problem Parameters
	x0_param = cp.Parameter(n)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)
	init_set = (cp.Parameter((m_init_set, n)), cp.Parameter(m_init_set))
	terminal_set = (cp.Parameter((m_terminal_set, n)), cp.Parameter(m_terminal_set))

	# Constraints
	init_constraint = [init_set[0] @ (x0_param - x[:,0]) <= init_set[1]]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraint = [terminal_set[0] @ x[:, N] <= terminal_set[1]]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += cp.quad_form((x[:,N] - state_reference[N]), P)
	cost = cp.Minimize(trajectory_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraint 
	problem = cp.Problem(cost, constraints)

	return x0_param, x, u, A, B, C, state_reference, input_reference, state_limits, input_limits, init_set, terminal_set, problem

def LTV_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, Q, R):
	n = Q.shape[0]
	m = R.shape[0]

	# Decision Variables
	x = cp.Variable((n, N + 1))
	u = cp.Variable((m, N))
	multipliers = cp.Variable(n_safe_set, nonneg=True) 

	# Problem Parameters
	x0_param = cp.Parameter(n)
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)

	# Constraints
	init_constraint = [x[:,0] == x0_param]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)	
	terminal_constraints = [safe_set @ multipliers == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	cost = cp.Minimize(trajectory_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints 
	problem = cp.Problem(cost, constraints)

	return x0_param, x, u, safe_set, value_function, A, B, C, state_reference, input_reference, state_limits, input_limits, problem

def LTV_tube_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, m_init_set, Q, R):
	# Setup
	n = Q.shape[0]
	m = R.shape[0]

	# Decision Variables
	x = cp.Variable((n, N + 1))
	u = cp.Variable((m, N))
	multipliers = cp.Variable(n_safe_set, nonneg=True) 

	# Problem Parameters
	x0_param = cp.Parameter(n)
	init_set = (cp.Parameter((m_init_set, n)), cp.Parameter(m_init_set))	
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)

	# Constraints
	init_constraint = [init_set[0] @ (x0_param - x[:,0]) <= init_set[1]]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)	
	terminal_constraints = [safe_set @ multipliers == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	cost = cp.Minimize(trajectory_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints 
	problem = cp.Problem(cost, constraints)

	return x0_param, init_set, x, u, safe_set, value_function, A, B, C, state_reference, input_reference, state_limits, input_limits, problem
