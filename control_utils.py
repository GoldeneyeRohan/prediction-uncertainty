import itertools
import polytope
import scipy.spatial
import numpy as np
import cvxpy as cp

def chull_to_poly(chull):
	"""
	Converts a scipy.spatial.ConvexHull object into a polytope.Polytope
	This function is convenient since the Polytope functions can be extremely
	slow if not pre-populated with both the vertices and equations.
	"""
	A = chull.equations[:,:-1]
	b = - chull.equations[:,-1]
	S = polytope.Polytope(A, b)
	S.vertices = chull.points[chull.vertices,:]
	return S

def poly_transform(P, A):
	"""
	Compute the linear transformation of the polytope P = {x | Ax <=b} with G:
	returns Polytope S = {y | \exists x \in P s.t. y = Gx} 
	"""
	vertices = polytope.extreme(P)
	transformed_vertices = vertices @ A.T

	if transformed_vertices.shape[1] > 1:
		chull = scipy.spatial.ConvexHull(transformed_vertices)
		S = chull_to_poly(chull)
	else:
		S = polytope.qhull(transformed_vertices)
	S = polytope.reduce(S)
	# S = polytope.qhull(transformed_vertices)
	return S

def poly_translate(P, z):
	"""
	Translate the polytope P by z:
	returns Polytope S = {y | \exists x \in P s.t. y = x + z}
	"""
	A = P.A
	b = P.b
	b_new = b + A @ z
	S = polytope.Polytope(A, b_new)
	if P.vertices is not None:
		S.vertices = P.vertices + z
	return S

def polytopic_support_function(P, x):
	"""
	Evaluate the support function of the polytope P = {x : Ax <= b} at x:
	h_P(x) = sup_y x^T y s.t. y \in P
	"""
	y = cp.Variable(len(x))
	prob = cp.Problem(cp.Maximize(x.T @ y), [P.A @ y <= P.b])
	prob.solve()
	h = prob.value
	y_opt = y.value
	return h, y_opt

def ellipsoidal_support_function(P, x):
	"""
	Evaluate the support function of the hyper-ellipsoid P = {x : x^TPx <= 1} at x:
	h_P(x) = sup_y x^Ty s.t. y \in P
	"""
	y = cp.Variable(len(x))
	prob = cp.Problem(cp.Maximize(x.T @ y), [cp.quad_form(y, P) <= 1])
	prob.solve()
	h = prob.value
	y_opt = y.value
	return h, y_opt


def minkowski_sum(A, B):
	"""
	Computes the Minkowski Sum of the Polytopes A and B using a projection algorithm
	returns Polytope S = {z | \exists x \in A, y \in B s.t. z = x + y}
	"""
	# Reduce Call Doesn't work in high dim even though scipy cvxhull is faster for other purposes...
	# vertices_a = polytope.extreme(A)
	# vertices_b = polytope.extreme(B)
	# vertex_combs = np.array(list(itertools.product(np.rollaxis(vertices_a, 0), np.rollaxis(vertices_b, 0))))
	# s_points = np.sum(vertex_combs, axis=1)
	# chull = scipy.spatial.ConvexHull(s_points)
	# S_chull = chull_to_poly(chull)
	
	As = np.vstack((np.hstack((B.A, - B.A)), np.hstack((np.zeros(A.A.shape), A.A))))
	bs = np.hstack((B.b, A.b))
	P = polytope.Polytope(As, bs)
	# P.vertices = np.hstack((S_chull.vertices, np.zeros((S_chull.vertices.shape[0], As.shape[1]))))
	S = P.project(np.arange(A.A.shape[1]) + 1)
	S = polytope.reduce(S)
	return S

def pontryagin_difference(A, B):
	"""
	Computes the Pontryagin Difference of the sets A and B using a support function algorithm
	- A is a Polytope instance
	- B is either a Polytope instance, or a square matrix defining the set {x : x^TBx <= 1}
	returns Polytope S = {x | x + y \in A \forall y \in B}
	"""
	if isinstance(B, polytope.Polytope):
		support_vec = np.array(list(zip(*[polytopic_support_function(B, a) for a in np.rollaxis(A.A, 0)]))[0])
	else:
		support_vec = np.array(list(zip(*[ellipsoidal_support_function(B, a) for a in np.rollaxis(A.A, 0)]))[0])
	S = polytope.Polytope(A.A, A.b - support_vec)
	return S

def autonomous_pre(P, A, B=None, K=None, U=None):
	"""
	Computes the Pre-set of P subject to the autonomous dynamics x_{t+1} = Ax_t
	returns Polytope S = Pre(P) = {x | Ax \in S}

	If B, K, U are not none, handles additional input constraints
	returns Polytope S = Pre(P) = {x | (A - BK)x, -Kx \in U}
	"""
	if B is None or K is None or U is None:
		S = polytope.Polytope(P.A @ A, P.b)
	else: 
		As = np.vstack((P.A @ (A - B @ K), - U.A @ K))
		bs = np.hstack((P.b, U.b))
		S = polytope.Polytope(As, bs)
		S = polytope.reduce(S)
	return S

def robust_autonomous_pre(P, W, A, B=None, K=None, U=None):
	"""
	Computes the robust Pre-set of P subject to the autonomous dynamics x_{t+1} = Ax_t + w_t, w_t \in W
	returns Polytope S = RPre(P) = {x | Ax + w \in S, \forall w \in W}

	If B, K, U are not none, handles additional input constraints
	returns Polytope S = RPre(P) = {x | (A - BK)x +w \in S Kx \in U, \forall w \in W}
	"""
	RP = pontryagin_difference(P, W)
	RPre = autonomous_pre(RP, A, B=B, K=K, U=U)
	return RPre

def is_invariant(P, A, B=None, K=None, U=None):
	"""
	Verifies whether P is an invariant set for the autonomous dynamics x_{t+1} = Ax_t
	returns True if P is invariant, otherwise False

	If B, K, U are not none, handles additional input constraints (A is closed loop otherwise)
	"""
	Pre = autonomous_pre(P, A, B=B, K=K, U=U)
	return polytope.is_subset(P, Pre)

def is_robust_invariant(P, W, A, B=None, K=None, U=None):
	"""
	Verifies whether P is a robust invariant set for the autonomous dynamics x_{t+1} = Ax_t + w, w \in W
	returns True if P is robust invariante, otherwise False

	If B, K, U are not none, handles additional input constraints (A is closed loop otherwise)
	"""
	RPre = robust_autonomous_pre(P, W, A, B=B, K=K, U=U)
	return polytope.is_subset(P, RPre)

def maximal_invariant(X, A, B=None, K=None, U=None):
	"""
	Computes the maximal invariant set of the constrainted autonomous dynamics x_{t+1} = (A - BK)x_t, x \in X, u \in U
	If B, K, U are None, then assumes A is closed dynamics.
	returns Polytope O s.t. any invariant S \subseteq O
	"""
	O_prev = X
	O_curr = autonomous_pre(O_prev, A, B=B, K=K, U=U).intersect(O_prev)
	while O_curr != O_prev:
		O_prev = O_curr
		O_curr = autonomous_pre(O_prev, A, B=B, K=K, U=U).intersect(O_prev)
	return O_curr


def minimal_invariant(A, W, n=15, epsilon=.01, max_k=200):
	"""
	Computes an approximation to the minimal invariant set of the autonomous dynamics
	x_{t+1} = Ax_t + w_t, w \in W
	returns Polytope M = (1 + k * epsilon) \sum_{i=0}^n A^iW, k = min integer s.t. M is invariant
	This approximation can be numerically unstable, you should first eyeball the i for which A^i ~ 0
	"""
	M_hat = W 
	for i in range(1, n + 1):
		AW = poly_transform(W, np.linalg.matrix_power(A, i))
		if AW.volume != 0:
			M_hat = minkowski_sum(M_hat, AW)
		else:
			break

	for k in range(1, max_k):
		if is_robust_invariant(M_hat, W, A):
			break
		M_hat.scale( (1 + k * epsilon) / (1 + (k - 1) * epsilon))

	return M_hat


def robust_maximal_invariant(X, W, M, A, B=None, K=None, U=None):
	"""
	Computes the robust maximal invariant set of the constrained autonomous dynamics
	x_{t+1} = (A - BK)x_t + w_t, x \in X, u \in U,  w \in W
	If B, K, U are None, then assumes A is closed loop dynamics matrix

	This function assumes that a minimal invarant M is known such that the dynamics are
	known to lie in the set x_t \in \bar{x}_t + M, where \bar{x}_{t+1} = (A + BK)\bar{x}_t 
	returns Polytope RO_nominal s.t. any robust invariant RS \subseteq RO for the nominal system
	and a Polytope RO_total s.t. RO_total = RO_nominal + M, the true maximal invariant
	"""
	X_bar = pontryagin_difference(X, M)
	if B is None or K is None or U is None:
		RO_nominal = maximal_invariant(X_bar, A)
	else:
		M_u = poly_transform(M, K)
		U_bar = pontryagin_difference(U, M_u)
		RO_nominal = maximal_invariant(X_bar, A, B=B, K=K, U=U_bar)
	RO_total = minkowski_sum(RO_nominal, M)	
	return RO_nominal, RO_total	

def compute_traj_cost(x_traj, u_traj, h):
	# import pdb; pdb.set_trace()
	cost_to_go = [h(x_traj[:,i], u_traj[:,i]) for i in range(x_traj.shape[1])]
	cost_to_go = np.cumsum([cost_to_go[::-1]])[::-1]
	return cost_to_go

def compute_nominal_traj(x_traj, u_traj, A, B, C, K):
	A_clp = A - B @ K
	e_traj = np.zeros(x_traj.shape)
	w_traj = np.zeros(x_traj[:,:-1].shape)
	
	for i in range(x_traj.shape[1] - 1):
		w_traj[:,i] = x_traj[:,i + 1] - (A @ x_traj[:,i] + B @ u_traj[:,i] + C)
		e_traj[:,i + 1] = A_clp @ e_traj[:,i] + w_traj[:,i]

	u_nominal = u_traj + K @ e_traj[:,:-1]
	x_nominal = x_traj - e_traj

	return x_nominal, u_nominal

def linearize_around(vehicle, x_traj, u_traj, dt):
	A, B, C = zip(*[vehicle.get_linearization(x_traj[:,i], u_traj[:,i], dt) for i in range(u_traj.shape[1])])
	return A, B, C	

def select_points(x, N, state_traj, input_traj, value_function):
	dists = np.linalg.norm(state_traj - x.reshape((len(x),1)), axis=0)
	min_dist_idx = np.argmin(dists)
	if min_dist_idx - int(N / 2) < 0 :
		i_min = 0
		i_max = N
	elif min_dist_idx + np.ceil(N / 2) >= len(dists):
		i_max = len(dists)
		i_min = i_max - N
	else:
		i_min = min_dist_idx - int(N / 2)
		i_max = min_dist_idx + np.ceil(N / 2)
	i_min = int(i_min)
	i_max = int(i_max)

	if i_max < state_traj.shape[1] - 1:
		successor_states = state_traj[:, i_min+1:i_max+1]
	else:
		successor_states = state_traj[:, i_min:i_max]

	return state_traj[:, i_min:i_max], input_traj[:, i_min:i_max], value_function[i_min:i_max], successor_states

def split_n(n, k):
	p = int(n / k)
	split = [p] * k
	split[0] += n - p * k
	return split

