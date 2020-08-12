import numpy as np 
import cvxpy as cp

def get_dynamics_constraints(x, u, A, B, C, N):
	dynamics_constraints = []
	for i in range(N):
		dynamics_constraints += [x[:, i+1] == A[i] @ x[:,i] + B[i] @ u[:,i] + C[i]]
	return dynamics_constraints
	
def get_state_constraints(x, slack, state_limits, N):
	state_constraints = [state_limits[0][0] @ x[:,0] <= state_limits[0][1] + slack[:,0]]
	for i in range(1, N):
		state_constraints += [state_limits[i][0] @ x[:,i] <= state_limits[i][1] + slack[:,i]]
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

def get_decision_variables(n, m, N, m_state_constraints):
	x = cp.Variable((n, N + 1))
	u = cp.Variable((m, N))
	slack = cp.Variable((m_state_constraints, N), nonneg=True)
	return x, u, slack

def format_slack_penalty(slack_penalty, m_state_constraints, N):
	if isinstance(slack_penalty, float):
		slack_penalty = slack_penalty * np.ones(m_state_constraints * N)
	else:
		slack_penalty = np.hstack([slack_penalty] * N)
	return slack_penalty

def format_terminal_slack_penalty(slack_penalty, m_constraints):
	if isinstance(slack_penalty, float):
		slack_penalty = slack_penalty * np.ones(m_constraints)
	return slack_penalty

def LTV_ftocp_solver(N, m_state_constraints, m_input_constraints, m_terminal_set, Q, R, P, slack_penalty=1e3, terminal_slack_penalty=1e3):
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
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, m_terminal_set)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(m_terminal_set, nonneg=True)

	# Problem Parameters
	x0_param = cp.Parameter(n)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)
	terminal_set = (cp.Parameter((m_terminal_set, n)), cp.Parameter(m_terminal_set))

	# Constraints
	init_constraint = [x[:,0] == x0_param]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraint = [terminal_set[0] @ x[:, N] <= terminal_set[1] + terminal_slack]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += cp.quad_form((x[:,N] - state_reference[N]), P)
	slack_cost = slack_penalty.T @ slack.flatten() + terminal_slack_penalty.T @ terminal_slack
	cost = cp.Minimize(trajectory_cost + slack_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraint 
	problem = cp.Problem(cost, constraints)

	return x0_param, x, u, slack, terminal_slack, A, B, C, state_reference, input_reference, state_limits, input_limits, terminal_set, problem

def LTV_tube_ftocp_solver(N, m_state_constraints, m_input_constraints, m_init_set, m_terminal_set, Q, R, P, slack_penalty=1e3, terminal_slack_penalty=1e3):
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, m_terminal_set)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(m_terminal_set, nonneg=True)
	
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
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraint = [terminal_set[0] @ x[:, N] <= terminal_set[1] + terminal_slack]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += cp.quad_form((x[:,N] - state_reference[N]), P)
	slack_cost = slack_penalty.T @ slack.flatten() + terminal_slack_penalty.T @ terminal_slack
	cost = cp.Minimize(trajectory_cost + slack_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraint 
	problem = cp.Problem(cost, constraints)

	return x0_param, x, u, slack, terminal_slack, A, B, C, state_reference, input_reference, state_limits, input_limits, init_set, terminal_set, problem

def LTV_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3):
	# Setup
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(n)
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
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)	
	terminal_constraints = [safe_set @ multipliers + terminal_slack == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	cost = cp.Minimize(trajectory_cost + slack_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints 
	problem = cp.Problem(cost, constraints)

	return x0_param, x, u, multipliers, slack, terminal_slack, safe_set, value_function, A, B, C, state_reference, input_reference, state_limits, input_limits, problem

def LTV_tube_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, m_init_set, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3):
	# Setup
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(n)
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
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)	
	terminal_constraints = [safe_set @ multipliers + terminal_slack == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	cost = cp.Minimize(trajectory_cost + slack_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints 
	problem = cp.Problem(cost, constraints)

	return x0_param, init_set, x, u, multipliers, slack, terminal_slack, safe_set, value_function, A, B, C, state_reference, input_reference, state_limits, input_limits, problem

def LBLMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, m_init_set, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3):
	# Setup
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	x_hat, _, _ = get_decision_variables(n, m, N, m_state_constraints)

	terminal_slack = cp.Variable(n)
	multipliers = cp.Variable(n_safe_set, nonneg=True) 
	terminal_slack_hat = cp.Variable(n)
	multipliers_hat = cp.Variable(n_safe_set, nonneg=True) 

	# Problem Parameters
	x0_param = cp.Parameter(n)
	init_set = (cp.Parameter((m_init_set, n)), cp.Parameter(m_init_set))	
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	A, B, C = get_dynamics_parameters(n, m, N)
	A_hat, B_hat, C_hat = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)

	# Constraints
	init_constraints = [init_set[0] @ (x0_param - x[:,0]) <= init_set[1]]
	init_constraints += [x[:,0] == x_hat[:,0]]

	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	dynamics_constraints += get_dynamics_constraints(x_hat, u, A_hat, B_hat, C_hat, N)

	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)

	terminal_constraints = [safe_set @ multipliers + terminal_slack == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]
	terminal_constraints += [safe_set @ multipliers_hat + terminal_slack_hat == x_hat[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers_hat == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers_hat
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	slack_cost += cp.quad_form(terminal_slack_hat, terminal_slack_penalty)
	cost = cp.Minimize(trajectory_cost + slack_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraints + state_constraints + input_constraints + terminal_constraints 
	problem = cp.Problem(cost, constraints)

	return (x0_param, init_set, x, u, multipliers, slack, x_hat, multipliers_hat, 
				terminal_slack, terminal_slack_hat, 
				safe_set, value_function, 
				A, B, C, A_hat, B_hat, C_hat, 
				state_reference, input_reference, state_limits, input_limits, problem)


def SCP_ftocp_solver(N, m_state_constraints, m_input_constraints, m_terminal_set, Q, R, P, slack_penalty=1e3, terminal_slack_penalty=1e3, regularization=1e2):
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, m_terminal_set)

	# Decision Variables
	dx, du, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(m_terminal_set, nonneg=True)

	# Problem Parameters
	x_param = cp.Parameter(dx.shape)
	u_param = cp.Parameter(du.shape)
	C = [np.zeros(n) for i in range(N)]
	A, B, _ = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)
	terminal_set = (cp.Parameter((m_terminal_set, n)), cp.Parameter(m_terminal_set))

	# Constraints
	init_constraint = [dx[:,0] == np.zeros(n)]
	dynamics_constraints = get_dynamics_constraints(dx, du, A, B, C, N)
	state_constraints = get_state_constraints(dx + x_param, slack, state_limits, N)
	input_constraints = get_input_constraints(du + u_param, input_limits, N)
	terminal_constraint = [terminal_set[0] @ (dx[:, N] + x_param[:,N]) <= terminal_set[1] + terminal_slack]

	# Cost function
	trajectory_cost = get_trajectory_cost(dx + x_param, du + u_param, Q, R, input_reference, state_reference, N)
	trajectory_cost += cp.quad_form((dx[:,N] + x_param[:,N] - state_reference[N]), P)
	slack_cost = slack_penalty.T @ slack.flatten() + terminal_slack_penalty.T @ terminal_slack
	regularization_cost = get_trajectory_cost(dx, du, np.eye(n) * regularization, np.eye(m) * regularization,
												[np.zeros(m) for _ in range(N)], [np.zeros(n) for _ in range(N + 1)], N)
	cost = cp.Minimize(trajectory_cost + slack_cost + regularization_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraint 
	problem = cp.Problem(cost, constraints)

	return x_param, u_param, dx, du, slack, terminal_slack, A, B, state_reference, input_reference, state_limits, input_limits, terminal_set, problem

def SCP_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3, regularization=1e2):
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	dx, du, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(n)
	multipliers = cp.Variable(n_safe_set, nonneg=True)

	# Problem Parameters
	x_param = cp.Parameter(dx.shape)
	u_param = cp.Parameter(du.shape)
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	C = [np.zeros(n) for i in range(N)]
	A, B, _ = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)

	# Constraints
	init_constraint = [dx[:,0] == np.zeros(n)]
	dynamics_constraints = get_dynamics_constraints(dx + x_param, du + u_param, A, B, C, N)
	state_constraints = get_state_constraints(dx + x_param, slack, state_limits, N)
	input_constraints = get_input_constraints(du + u_param, input_limits, N)
	terminal_constraints = [safe_set @ multipliers + terminal_slack == x_param[:,-1] + dx[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(dx + x_param, du + u_param, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	regularization_cost = get_trajectory_cost(dx, du, np.eye(n) * regularization, np.eye(m) * regularization,
												[np.zeros(m) for _ in range(N)], [np.zeros(n) for _ in range(N + 1)], N)
	cost = cp.Minimize(trajectory_cost + slack_cost + regularization_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints
	problem = cp.Problem(cost, constraints)

	return x_param, u_param, dx, du, multipliers, slack, terminal_slack, safe_set, value_function, A, B, state_reference, input_reference, state_limits, input_limits, problem


def alternate_SCP_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3, regularization=1e2):
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	terminal_slack = cp.Variable(n)
	multipliers = cp.Variable(n_safe_set, nonneg=True)

	# Problem Parameters
	x_param = cp.Parameter(x.shape)
	u_param = cp.Parameter(u.shape)
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)

	# Constraints
	init_constraint = [x[:,0] == x_param[:,0]]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraints = [safe_set @ multipliers + terminal_slack == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	regularization_cost = get_trajectory_cost(x - x_param, u - u_param, np.eye(n) * regularization, np.eye(m) * regularization,
												[np.zeros(m) for _ in range(N)], [np.zeros(n) for _ in range(N + 1)], N)
	cost = cp.Minimize(trajectory_cost + slack_cost + regularization_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints
	problem = cp.Problem(cost, constraints)

	return x_param, u_param, x, u, multipliers, slack, terminal_slack, safe_set, value_function, A, B, C, state_reference, input_reference, state_limits, input_limits, problem


def uncertain_SCP_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3, regularization=1e2, uncertainty_cost=1e3):
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	z = cp.vstack([x[:,:-1], u, np.ones((1, u.shape[1]))])

	terminal_slack = cp.Variable(n)
	multipliers = cp.Variable(n_safe_set, nonneg=True)

	# Problem Parameters
	x_param = cp.Parameter(x.shape)
	u_param = cp.Parameter(u.shape)
	z_param = cp.vstack([x_param[:,:-1], u_param, np.ones((1, u_param.shape[1]))])
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	safe_set_uncertainty = cp.Parameter(n_safe_set)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)
	uncertainties = [cp.Parameter((n + m + 1, n + m + 1), PSD=True) for _ in range(N)]

	# Constraints
	init_constraint = [x[:,0] == x_param[:,0]]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraints = [safe_set @ multipliers + terminal_slack == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	uncertainty_penalty = 0
	for i in range(N):
		# uncertainty_penalty += cp.quad_form(z[:,i] - z_param[:,i], uncertainties[i]) * regularization
		uncertainty_penalty += cp.quad_form(z[:,i], uncertainties[i]) * uncertainty_cost
	uncertainty_penalty += multipliers @ safe_set_uncertainty * uncertainty_cost
	# import pdb; pdb.set_trace()
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	regularization_cost = get_trajectory_cost(x - x_param, u - u_param, np.eye(n) * regularization, np.eye(m) * regularization,
												[np.zeros(m) for _ in range(N)], [np.zeros(n) for _ in range(N + 1)], N)
	cost = cp.Minimize(trajectory_cost + slack_cost  + uncertainty_penalty + regularization_cost)

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints
	problem = cp.Problem(cost, constraints)

	return x_param, u_param, x, u, multipliers, slack, terminal_slack, safe_set, value_function, uncertainties, safe_set_uncertainty, A, B, C, state_reference, input_reference, state_limits, input_limits, problem


def cov_SCP_LMPC_ftocp_solver(N, n_safe_set, m_state_constraints, m_input_constraints, Q, R, slack_penalty=1e3, terminal_slack_penalty=1e3, regularization=1e2, uncertainty_cost=1e3):
	n = Q.shape[0]
	m = R.shape[0]
	slack_penalty = format_slack_penalty(slack_penalty, m_state_constraints, N)
	terminal_slack_penalty = format_terminal_slack_penalty(terminal_slack_penalty, n)
	terminal_slack_penalty = np.diag(terminal_slack_penalty)

	# Decision Variables
	x, u, slack = get_decision_variables(n, m, N, m_state_constraints)
	z = cp.vstack([x[:,:-1], u, np.ones((1, u.shape[1]))])

	terminal_slack = cp.Variable(n)
	multipliers = cp.Variable(n_safe_set, nonneg=True)

	# Problem Parameters
	x_param = cp.Parameter(x.shape)
	u_param = cp.Parameter(u.shape)
	z_param = cp.vstack([x_param[:,:-1], u_param, np.ones((1, u_param.shape[1]))])
	safe_set = cp.Parameter((n, n_safe_set))
	value_function = cp.Parameter(n_safe_set)
	safe_set_uncertainty = cp.Parameter(n_safe_set)
	A, B, C = get_dynamics_parameters(n, m, N)
	state_reference, input_reference = get_reference_parameters(n, m, N)
	state_limits, input_limits = get_constraint_parameters(n, m, m_state_constraints, m_input_constraints, N)
	uncertainties = [cp.Parameter((n + m + 1, n + m + 1), PSD=True) for _ in range(N)]

	# Constraints
	init_constraint = [x[:,0] == x_param[:,0]]
	dynamics_constraints = get_dynamics_constraints(x, u, A, B, C, N)
	state_constraints = get_state_constraints(x, slack, state_limits, N)
	input_constraints = get_input_constraints(u, input_limits, N)
	terminal_constraints = [safe_set @ multipliers + terminal_slack == x[:,-1]]
	terminal_constraints += [np.ones(n_safe_set) @ multipliers == 1]

	# Cost function
	trajectory_cost = get_trajectory_cost(x, u, Q, R, input_reference, state_reference, N)
	trajectory_cost += value_function @ multipliers
	uncertainty_penalty = 0
	for i in range(N):
		uncertainty_penalty += cp.quad_form(z[:,i] - z_param[:,i], uncertainties[i]) * regularization
		uncertainty_penalty += cp.quad_form(z[:,i], uncertainties[i]) * uncertainty_cost
	uncertainty_penalty += multipliers @ safe_set_uncertainty * uncertainty_cost
	# import pdb; pdb.set_trace()
	slack_cost = slack_penalty.T @ slack.flatten() + cp.quad_form(terminal_slack, terminal_slack_penalty)
	# regularization_cost = get_trajectory_cost(x - x_param, u - u_param, np.eye(n) * regularization, np.eye(m) * regularization,
												# [np.zeros(m) for _ in range(N)], [np.zeros(n) for _ in range(N + 1)], N)
	cost = cp.Minimize(trajectory_cost + slack_cost  + uncertainty_penalty) #+ regularization_cost

	# Solve and Return
	constraints = dynamics_constraints + init_constraint + state_constraints + input_constraints + terminal_constraints
	problem = cp.Problem(cost, constraints)

	return x_param, u_param, x, u, multipliers, slack, terminal_slack, safe_set, value_function, uncertainties, safe_set_uncertainty, A, B, C, state_reference, input_reference, state_limits, input_limits, problem

