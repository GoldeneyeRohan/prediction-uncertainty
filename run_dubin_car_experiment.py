import controllers
import controlpy
import control_utils
import cvxpy as cp
import dynamics_models
import numpy as np
import polytope
import system_id
import tqdm
import uncertainty_utils
import experiment_utils
import trajectory_optimizers
import sys


#### ------------------------------------------------------ SETUP ---------------------------------------###

#### ------------------------------------------------------ SETUP ---------------------------------------###
# Dynamics Parameters
n_states = 4
n_inputs = 2
dt = 0.1
init_state = np.array([-3.5,0, np.pi/2, 0])
linearization_state = np.zeros(n_states)
linearization_input = np.zeros(n_inputs)

# State and Input Constraints
delta_lim = np.pi / 6
a_lim = 0.3
input_limits = polytope.box2poly([[-a_lim, a_lim], [-delta_lim, delta_lim]])
state_limits = polytope.box2poly([[-4,1],[-1,1],[-5 * np.pi, 5 *np.pi],[-0.1,1]])
state_constraints = (state_limits.A, state_limits.b)
input_constraints = (input_limits.A, input_limits.b)
input_bounds = (-np.array([a_lim, delta_lim]), np.array([a_lim, delta_lim]))

# Control Task
Q = np.diag([1, 10, 1, 1]) 
R = np.eye(n_inputs)
stage_cost = lambda x, u: x.T @ Q @ x + u.T @ R @ u

num_init_episodes = 10
init_input_noise_var = 1e-3
init_input_noise = init_input_noise_var * np.eye(n_inputs)
t_inp = uncertainty_utils.calc_t(0.95, n_inputs)
idb = t_inp * np.sqrt(init_input_noise_var)
input_noise_bound = polytope.box2poly([[-idb, idb]] * n_inputs)

# Noise and Disturbances
process_noise_var = 1e-5
process_noise = process_noise_var * np.eye(n_states)
t = uncertainty_utils.calc_t(0.95, n_states)
db = t * np.sqrt(process_noise_var)
disturbance_bound = polytope.box2poly([[-db, db]] * n_states)

# Experiment Details
episode_length = int(10 / dt)
controller_horizon = 5
state_reference = np.zeros(n_states)
input_reference = np.zeros(n_inputs)
num_episodes = 15

# System ID
h = 2
lamb = 1e-2
n_sysid_pts = 500
n_sysid_it = 10

def model_callback(model, controller, episode_length):
#     As, Bs, Cs, covs, errors = model.regress_models(controller.x_traj.value.T, controller.u_traj.value.T)
    u_ss = controller.input_safe_set @ controller.multipliers.value
    input_traj = np.vstack((controller.u_traj[:,1:].value.T, u_ss))
    As, Bs, Cs, covs, errors = model.regress_models(controller.x_traj[:,1:].value.T, input_traj)
    return As, Bs, Cs, covs, errors


# Save Results
save_dir = "dubin_car_expts_scp_lmpc/"
save_data = True
save_number = int(sys.argv[1])

# Vehicle
def get_vehicle():
    init_state_noisy = np.random.multivariate_normal(init_state, process_noise)
    vehicle = dynamics_models.DubinCar(init_state_noisy, dt, process_noise, use_ode_integrator=False)
    return vehicle

def sim_traj(vehicle, controller, input_limits, episode_length=episode_length, solver_helpers=None, model=None, model_callback=None, input_noise=None, ):
    x_traj = [vehicle.x]
    u_traj = []
    
    x_pred_trajs = []
    u_pred_trajs = []
    model_covs = []
    
    slacks = []
    terminal_slacks = []
    
    for _ in tqdm.tqdm(range(episode_length)):
        u_command = controller.solve(x_traj[-1]) if solver_helpers is None else controller.solve(x_traj[-1], solver_helpers[0], solver_helpers[1])
        if u_command is None:
            print("controller error at iteration %d" %_)
            print("state:")
            print(x_traj[-1])
            break
        if input_noise is not None: 
            u_noise = np.random.multivariate_normal(np.zeros(input_limits[0].shape), input_noise)
            u_command = u_command + u_noise
        if model is not None:
            As, Bs, Cs, covs, errors = model_callback(model, controller, episode_length)
            controller.set_models(As, Bs, Cs)
            model_covs.append(covs)
        u = np.minimum(np.maximum(input_limits[0], u_command), input_limits[1])
        x_next = vehicle.f(u)

        x_traj.append(x_next)
        u_traj.append(u)
        if hasattr(controller, "pred_uncertainty"):
            model_covs.append(controller.pred_uncertainty)
        if hasattr(controller, "x_traj"):
            if isinstance(controller.x_traj, cp.Variable):
                x_pred_trajs.append(controller.x_traj.value.T)
                u_pred_trajs.append(controller.u_traj.value.T)
            else:
                x_pred_trajs.append(controller.x_traj.T)
                u_pred_trajs.append(controller.u_traj.T)
                
            slacks.append(controller.slack.value)
            terminal_slacks.append(controller.terminal_slack.value)
        
    return np.array(x_traj), np.array(u_traj), x_pred_trajs, u_pred_trajs, slacks, terminal_slacks, model_covs

#### ------------------------------------------------------ INIT ---------------------------------------####
print("CONTROLLER INITIALIZATION")
#### ------------------------------------------------------ INIT ---------------------------------------####

vehicle = get_vehicle()
A, B, C = vehicle.get_linearization(linearization_state, linearization_input, dt)
A[1,2] = 0.025

init_state_limits = control_utils.pontryagin_difference(state_limits, disturbance_bound)
init_input_limits = control_utils.pontryagin_difference(input_limits, input_noise_bound)
init_state_constraints = (init_state_limits.A, init_state_limits.b)
init_input_constraints = (init_input_limits.A, init_input_limits.b)

controller = controllers.LTI_MPC_Tracker(A, B, C,
                                         controller_horizon * 2, 
                                         Q, 3*R, state_reference, input_reference, 
                                         init_state_constraints, init_input_constraints)
controller.build()

x_init_trajs = []
u_init_trajs = []
for _ in range(num_init_episodes):
    vehicle = get_vehicle()
    x_init_traj, u_init_traj, x_init_preds, u_init_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, input_noise=init_input_noise)
    x_init_trajs.append(x_init_traj)
    u_init_trajs.append(u_init_traj)
init_value_functions = [control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost) for x_traj, u_traj in zip(x_init_trajs, u_init_trajs)]

# filename = "tracking_mpc"
# # if save_data:
# #     experiment_utils.save_result(save_dir, filename, x_init_trajs, u_init_trajs, None, None, None, i=save_number)


# #### ------------------------------------------------------ NAIVE ---------------------------------------####
# print("NAIVE TV LMPC")
# #### ------------------------------------------------------ NAIVE ---------------------------------------####
# controller = controllers.LTV_LMPC(A, B, C,
#                                      controller_horizon, 
#                                      Q,R, state_reference, input_reference, 
#                                      state_constraints, input_constraints)

# for x_init_traj, u_init_traj, init_value_func in zip(x_init_trajs, u_init_trajs, init_value_functions):
#     controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, init_value_func)

# model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
# for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
#     model.add_trajectory(x_init_traj, u_init_traj)

# slack_per_episode = []
# term_slack_per_episode = []
# model_covs_per_episode = []
# x_preds_per_episode = []
# for episode in range(num_episodes):
#     vehicle = get_vehicle()
    
#     x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
#                                                                                      model=model, model_callback=model_callback)
    
#     value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
#     controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
#     model.add_trajectory(x_traj, u_traj)
    
#     As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
#     controller.set_models(As, Bs, Cs)
    
#     slack_per_episode.append(slacks)
#     term_slack_per_episode.append(terminal_slacks)
#     model_covs_per_episode.append(model_covs)
#     x_preds_per_episode.append(x_preds)


# filename = "naive_ltv_lmpc"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode, i=save_number)

# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# print("LEARNING BASED LEARNING MPC")
# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# controller = controllers.LBLMPC(A, B, C,
#                                     controller_horizon, 
#                                     Q, R, state_reference, input_reference, 
#                                     state_constraints, input_constraints, (disturbance_bound.A, disturbance_bound.b))
# ## Minimal Invariant Cannot Be Computed
# alpha = 2
# M = polytope.box2poly([[-db * alpha, db * alpha]] * n_states)
# X = polytope.Polytope(*controller.state_constraints)
# U = polytope.Polytope(*controller.input_constraints)
# X_bar = control_utils.pontryagin_difference(X, M)
# U_bar = control_utils.pontryagin_difference(U, control_utils.poly_transform(M, controller.K))
# controller.state_constraints = (X_bar.A, X_bar.b)
# controller.input_constraints = (U_bar.A, U_bar.b)
# controller.init_constraint = (M.A, M.b)

# for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
#     x_nominal, u_nominal = control_utils.compute_nominal_traj(x_init_traj.T, u_init_traj.T, A, B, C, controller.K)
#     controller.add_trajectory(x_nominal[:,:-1], u_nominal, value_function)

# model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
# for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
#     model.add_trajectory(x_init_traj, u_init_traj)

# slack_per_episode = []
# term_slack_per_episode = []
# model_covs_per_episode = []
# x_preds_per_episode = []
# for episode in range(num_episodes):
#     vehicle = get_vehicle()
#     x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
#                                                                                      model=model, model_callback=model_callback)
    
#     x_nominal, u_nominal = control_utils.compute_nominal_traj(x_traj.T, u_traj.T, A, B, C, controller.K)
#     value_function = control_utils.compute_traj_cost(x_nominal[:,:-1], u_nominal, stage_cost)

#     controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
#     model.add_trajectory(x_traj, u_traj)
    
#     As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
#     controller.set_models(As, Bs, Cs)
    
#     slack_per_episode.append(slacks)
#     term_slack_per_episode.append(terminal_slacks)
#     model_covs_per_episode.append(model_covs)
#     x_preds_per_episode.append(x_preds)

# filename = "lblmpc"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode, i=save_number)

# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# print("LTV LMPC WITH LOCAL SAFE SETS")
# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# safe_set_size = 30
# n_safe_set_it = 3
# controller = controllers.Local_LTV_LMPC(A, B, C,
#                                      controller_horizon, 
#                                      Q, R, state_reference, input_reference, 
#                                      state_constraints, input_constraints, n_safe_set_it, safe_set_size)
# controller.build()
# for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
#     controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

# model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
# for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
#     model.add_trajectory(x_init_traj, u_init_traj)

# slack_per_episode = []
# term_slack_per_episode = []
# model_covs_per_episode = []
# x_preds_per_episode = []
# for episode in range(num_episodes):
#     vehicle = get_vehicle()
    
#     x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
#                                                                                      model=model, model_callback=model_callback)
    
#     value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
#     controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
#     model.add_trajectory(x_traj, u_traj)
    
#     As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
#     controller.set_models(As, Bs, Cs)
    
#     slack_per_episode.append(slacks)
#     term_slack_per_episode.append(terminal_slacks)
#     model_covs_per_episode.append(model_covs)
#     x_preds_per_episode.append(x_preds)

# filename = "local_ltv_lmpc"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode,i=save_number)

# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# print("TRUE LTV LMPC")
# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# controller = controllers.True_LTV_LMPC(A, B, C,
#                                      controller_horizon, 
#                                      Q, R, state_reference, input_reference, 
#                                      state_constraints, input_constraints)
# for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
#     controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

# model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb,  n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
# for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
#     model.add_trajectory(x_init_traj, u_init_traj)
    
# def ltv_model_callback(vehicle, controller, episode_length):
#     i = controller.i
#     N = controller.N
#     i = min(episode_length - N, i)
#     x_traj = controller.traj_list[-1][:,i:i+N]
#     u_traj = controller.input_traj_list[-1][:,i:i+N]
#     As, Bs, Cs, covs, errors = model.regress_models(controller.x_traj.value.T, controller.u_traj.value.T)
#     return As, Bs, Cs, covs, errors

# slack_per_episode = []
# term_slack_per_episode = []
# model_covs_per_episode = []
# x_preds_per_episode = []
# for episode in range(num_episodes):
#     vehicle = get_vehicle()
    
#     x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
#                                                                                      model=model, model_callback=ltv_model_callback)
    
#     value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
#     controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
#     model.add_trajectory(x_traj, u_traj)
    
#     As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
#     controller.set_models(As, Bs, Cs)
    
#     slack_per_episode.append(slacks)
#     term_slack_per_episode.append(terminal_slacks)
#     model_covs_per_episode.append(model_covs)
#     x_preds_per_episode.append(x_preds)

# filename = "true_ltv_lmpc"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode,i=save_number)

# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# print("LTI LMPC")
# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# controller = controllers.LTI_LMPC(A, B, C,
#                                      controller_horizon, 
#                                      Q, R, state_reference, input_reference, 
#                                      state_constraints, input_constraints)
# for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
#     controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

# slack_per_episode = []
# term_slack_per_episode = []
# x_preds_per_episode = []
# for episode in range(num_episodes):
#     vehicle = get_vehicle()
#     x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds)
#     value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
#     controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
#     slack_per_episode.append(slacks)
#     term_slack_per_episode.append(terminal_slacks)
#     x_preds_per_episode.append(x_preds)

# filename = "lti_lmpc"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, None, i=save_number)

# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# print("LTV TUBE LMPC")
# #### ------------------------------------------------------ LBLMPC ---------------------------------------####
# ## Minimal Invariant Cannot Be Computed
# alpha = 3
# M = polytope.box2poly([[-db * alpha, db * alpha]] * n_states)
# minimal_invariant = (M.A, M.b)

# controller = controllers.LTV_Tube_LMPC(A, B, C,
#                                     controller_horizon, 
#                                     Q, R, state_reference, input_reference, 
#                                     state_constraints, input_constraints, minimal_invariant)

# for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
#     controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

# model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
# for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
#     model.add_trajectory(x_init_traj, u_init_traj)

# slack_per_episode = []
# term_slack_per_episode = []
# model_covs_per_episode = []
# x_preds_per_episode = []
# for episode in range(num_episodes):
#     vehicle = get_vehicle()
    
#     x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
#                                                                                      model=model, model_callback=model_callback)
    
#     value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
#     controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
#     model.add_trajectory(x_traj, u_traj)
    
#     As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
#     controller.set_models(As, Bs, Cs)
    
#     slack_per_episode.append(slacks)
#     term_slack_per_episode.append(terminal_slacks)
#     model_covs_per_episode.append(model_covs)
#     x_preds_per_episode.append(x_preds)

# filename = "tube_ltv_lmpc"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode,i=save_number)

###### ---------------------------------------------------- SCP ----------------------------------------------- ##########
# model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
# for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
#     model.add_trajectory(x_init_traj, u_init_traj)

# def estimate_traj(model, controller, init_state, episode_length):
#     x_sim = [init_state]
#     u_sim = []
#     for i in range(episode_length):
#         u = controller.solve(x_sim[-1])
#         x, cov = model.predict(x_sim[-1].reshape((1, n_states)), u.reshape((1, n_inputs)))
#         x = x[0]
#         u_sim.append(u)
#         x_sim.append(x)
#     return np.array(x_sim), np.array(u_sim)
    
# def optimize_trajectory(traj_opt, model, x_init_traj, u_init_traj):
#     for i in tqdm.tqdm(range(20)):
#         vehicle = get_vehicle()
#         if i == 0:
#             As, Bs, Cs, covs, error = model.regress_models(x_init_traj, u_init_traj)
#             x_traj_opt, u_traj_opt, converged = traj_opt.solve_iteration(x_init_traj.T, u_init_traj.T, As, Bs)
#         else:
#             As, Bs, Cs, covs, error = model.regress_models(x_traj, u_traj)
#             x_traj_opt, u_traj_opt, converged = traj_opt.solve_iteration(x_traj.T, u_traj.T, As, Bs)

#         controller = traj_opt.get_controller()
#         x_traj, u_traj = estimate_traj(model, controller, x_init_traj[0,:], episode_length)
#         if converged:
#             break

# x_trajs_per_episode = [x_init_traj]
# u_trajs_per_episode = [u_init_traj]
# value_functions_per_episode = [init_value_functions[-1]]
# for episode in range(num_episodes):
#     traj_opt = trajectory_optimizers.SCP_Traj_Opt(episode_length, Q, R, state_reference, input_reference, state_constraints, input_constraints, tolerance=1e-5, regularization=1e0)
#     traj_opt.build()
    
#     optimize_trajectory(traj_opt, model, x_trajs_per_episode[-1], u_trajs_per_episode[-1])
#     vehicle = get_vehicle()#dynamics_models.DubinCar(init_state, dt, process_noise * 0, use_ode_integrator=False)
#     controller = traj_opt.get_controller()
#     x_traj, u_traj, _, _, _, _, _ = sim_traj(vehicle, controller, input_bounds)
    
#     value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
#     value_functions_per_episode.append(value_function)
#     model.add_trajectory(x_traj, u_traj)
    
#     x_trajs_per_episode.append(x_traj)
#     u_trajs_per_episode.append(u_traj)

# filename = "open_loop_scp"
# if save_data:
#     experiment_utils.save_result(save_dir, filename, [x.T for x in x_trajs_per_episode], [u.T for u in u_trajs_per_episode], None, value_functions_per_episode, None, i=save_number)

#########----------------------------------- UNCERTAIN SCP LMPC ------------------------------------------------------------------------------------------------------------------------
###############__________________________________--------------------------------------__________________________________________________________________________________________________________

model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)

for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
    model.add_trajectory(x_init_traj, u_init_traj)

def get_linearizations(x_traj, u_traj):
    As, Bs, Cs, covs, errors = model.regress_models(x_traj.T, u_traj.T)
    return As, Bs, Cs, covs

def estimate_traj(x0, u_traj):
    x_sim = [x0]
    for u in np.rollaxis(u_traj, 1):
        x, cov = model.predict(x_sim[-1].reshape((1, n_states)), u.reshape((1, n_inputs)))
        x_sim.append(x)
#     import pdb; pdb.set_trace()
    return np.vstack(x_sim).T, np.copy(u_traj)

solver_helpers = (get_linearizations, estimate_traj)

safe_set_size = 30
n_safe_set_it = 3
n_iter = 5
regularizations = [1e-20, 1e2, 1e-20, 1e2]
uncertainty_costs = [1e-20, 1e-20, 1e0, 1e0]
names = ["no_reg_no_uncert", "reg", "uncert", "reg_and_uncert"]

for reg, uncert, name in zip(regularizations, uncertainty_costs, names):
    print(name)
    controller = controllers.Uncertain_Local_SCP_LMPC(controller_horizon, Q, R, 
                                                state_reference, input_reference, 
                                                state_constraints, input_constraints, 
                                                n_safe_set_it, safe_set_size, 
                                                n_iter=n_iter, tolerance=1e-3, regularization=reg, uncertainty_cost=uncert)     
    for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
        controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)
    controller.build()

    slack_per_episode = []
    term_slack_per_episode = []
    preds_per_episode = []
    model_covs_per_episode = []
    for episode in range(num_episodes):
        vehicle = get_vehicle()
    #     import pdb; pdb.set_trace()
        x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, episode_length, solver_helpers=solver_helpers)
        value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
        model.add_trajectory(x_traj, u_traj)
        controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
        slack_per_episode.append(slacks)
        term_slack_per_episode.append(terminal_slacks)
        preds_per_episode.append(x_preds)
        model_covs_per_episode.append(model_covs)


    filename = name
    if save_data:
        experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, preds_per_episode, controller.value_func_list, model_covs_per_episode, i=save_number)

print("FINISHED EXPERIMENT")
