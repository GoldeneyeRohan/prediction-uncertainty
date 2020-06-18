import controllers
import controlpy
import control_utils
import dynamics_models
import numpy as np
import polytope
import system_id
import tqdm
import uncertainty_utils
import experiment_utils
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
num_episodes = 15

# System ID
h = 2
lamb = 1e-2
n_sysid_pts = 500
n_sysid_it = 10

def model_callback(model, controller, episode_length):
    As, Bs, Cs, covs, errors = model.regress_models(controller.x_traj.value.T, controller.u_traj.value.T)
    return As, Bs, Cs, covs, errors

# Save Results
save_dir = "dubin_car_expts_2/"
save_data = True
save_number = int(sys.argv[1])

# Vehicle
def get_vehicle():
    init_state_noisy = np.random.multivariate_normal(init_state, process_noise)
    vehicle = dynamics_models.DubinCar(init_state_noisy, dt, process_noise, use_ode_integrator=False)
    return vehicle

def sim_traj(vehicle, controller, input_limits, episode_length=episode_length, model=None, model_callback=None, input_noise=None):
    x_traj = [vehicle.x]
    u_traj = []
    
    x_pred_trajs = []
    u_pred_trajs = []
    model_covs = []
    
    slacks = []
    terminal_slacks = []
    
    for _ in tqdm.tqdm(range(episode_length)):
        u_command = controller.solve(x_traj[-1])
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
        
        x_pred_trajs.append(controller.x_traj.value.T)
        u_pred_trajs.append(controller.u_traj.value.T)
        
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
                                         Q, 3*R, state_reference, 
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

filename = "tracking_mpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, x_init_trajs, u_init_trajs, None, None, None, i=save_number)


#### ------------------------------------------------------ NAIVE ---------------------------------------####
print("NAIVE TV LMPC")
#### ------------------------------------------------------ NAIVE ---------------------------------------####
controller = controllers.LTV_LMPC(A, B, C,
                                     controller_horizon, 
                                     Q,R, state_reference, 
                                     state_constraints, input_constraints)

for x_init_traj, u_init_traj, init_value_func in zip(x_init_trajs, u_init_trajs, init_value_functions):
    controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, init_value_func)

model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
    model.add_trajectory(x_init_traj, u_init_traj)

slack_per_episode = []
term_slack_per_episode = []
model_covs_per_episode = []
x_preds_per_episode = []
for episode in range(num_episodes):
    vehicle = get_vehicle()
    
    x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
                                                                                     model=model, model_callback=model_callback)
    
    value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
    controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
    model.add_trajectory(x_traj, u_traj)
    
    As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
    controller.set_models(As, Bs, Cs)
    
    slack_per_episode.append(slacks)
    term_slack_per_episode.append(terminal_slacks)
    model_covs_per_episode.append(model_covs)
    x_preds_per_episode.append(x_preds)


filename = "naive_ltv_lmpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode, i=save_number)

#### ------------------------------------------------------ LBLMPC ---------------------------------------####
print("LEARNING BASED LEARNING MPC")
#### ------------------------------------------------------ LBLMPC ---------------------------------------####
controller = controllers.LBLMPC(A, B, C,
                                    controller_horizon, 
                                    Q, R, state_reference, 
                                    state_constraints, input_constraints, (disturbance_bound.A, disturbance_bound.b))
## Minimal Invariant Cannot Be Computed
alpha = 2
M = polytope.box2poly([[-db * alpha, db * alpha]] * n_states)
X = polytope.Polytope(*controller.state_constraints)
U = polytope.Polytope(*controller.input_constraints)
X_bar = control_utils.pontryagin_difference(X, M)
U_bar = control_utils.pontryagin_difference(U, control_utils.poly_transform(M, controller.K))
controller.state_constraints = (X_bar.A, X_bar.b)
controller.input_constraints = (U_bar.A, U_bar.b)
controller.init_constraint = (M.A, M.b)

for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
    x_nominal, u_nominal = control_utils.compute_nominal_traj(x_init_traj.T, u_init_traj.T, A, B, C, controller.K)
    controller.add_trajectory(x_nominal[:,:-1], u_nominal, value_function)

model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
    model.add_trajectory(x_init_traj, u_init_traj)

slack_per_episode = []
term_slack_per_episode = []
model_covs_per_episode = []
x_preds_per_episode = []
for episode in range(num_episodes):
    vehicle = get_vehicle()
    x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
                                                                                     model=model, model_callback=model_callback)
    
    x_nominal, u_nominal = control_utils.compute_nominal_traj(x_traj.T, u_traj.T, A, B, C, controller.K)
    value_function = control_utils.compute_traj_cost(x_nominal[:,:-1], u_nominal, stage_cost)

    controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
    model.add_trajectory(x_traj, u_traj)
    
    As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
    controller.set_models(As, Bs, Cs)
    
    slack_per_episode.append(slacks)
    term_slack_per_episode.append(terminal_slacks)
    model_covs_per_episode.append(model_covs)
    x_preds_per_episode.append(x_preds)

filename = "lblmpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode, i=save_number)

#### ------------------------------------------------------ LBLMPC ---------------------------------------####
print("LTV LMPC WITH LOCAL SAFE SETS")
#### ------------------------------------------------------ LBLMPC ---------------------------------------####
safe_set_size = 30
n_safe_set_it = 3
controller = controllers.Local_LTV_LMPC(A, B, C,
                                     controller_horizon, 
                                     Q, R, state_reference, 
                                     state_constraints, input_constraints, n_safe_set_it, safe_set_size)
controller.build()
for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
    controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
    model.add_trajectory(x_init_traj, u_init_traj)

slack_per_episode = []
term_slack_per_episode = []
model_covs_per_episode = []
x_preds_per_episode = []
for episode in range(num_episodes):
    vehicle = get_vehicle()
    
    x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
                                                                                     model=model, model_callback=model_callback)
    
    value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
    controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
    model.add_trajectory(x_traj, u_traj)
    
    As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
    controller.set_models(As, Bs, Cs)
    
    slack_per_episode.append(slacks)
    term_slack_per_episode.append(terminal_slacks)
    model_covs_per_episode.append(model_covs)
    x_preds_per_episode.append(x_preds)

filename = "local_ltv_lmpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode,i=save_number)

#### ------------------------------------------------------ LBLMPC ---------------------------------------####
print("TRUE LTV LMPC")
#### ------------------------------------------------------ LBLMPC ---------------------------------------####
controller = controllers.True_LTV_LMPC(A, B, C,
                                     controller_horizon, 
                                     Q, R, state_reference, 
                                     state_constraints, input_constraints)
for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
    controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb,  n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
    model.add_trajectory(x_init_traj, u_init_traj)
    
def ltv_model_callback(vehicle, controller, episode_length):
    i = controller.i
    N = controller.N
    i = min(episode_length - N, i)
    x_traj = controller.traj_list[-1][:,i:i+N]
    u_traj = controller.input_traj_list[-1][:,i:i+N]
    As, Bs, Cs, covs, errors = model.regress_models(controller.x_traj.value.T, controller.u_traj.value.T)
    return As, Bs, Cs, covs, errors

slack_per_episode = []
term_slack_per_episode = []
model_covs_per_episode = []
x_preds_per_episode = []
for episode in range(num_episodes):
    vehicle = get_vehicle()
    
    x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
                                                                                     model=model, model_callback=ltv_model_callback)
    
    value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
    controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
    model.add_trajectory(x_traj, u_traj)
    
    As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
    controller.set_models(As, Bs, Cs)
    
    slack_per_episode.append(slacks)
    term_slack_per_episode.append(terminal_slacks)
    model_covs_per_episode.append(model_covs)
    x_preds_per_episode.append(x_preds)

filename = "true_ltv_lmpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode,i=save_number)

#### ------------------------------------------------------ LBLMPC ---------------------------------------####
print("LTI LMPC")
#### ------------------------------------------------------ LBLMPC ---------------------------------------####
controller = controllers.LTI_LMPC(A, B, C,
                                     controller_horizon, 
                                     Q, R, state_reference, 
                                     state_constraints, input_constraints)
for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
    controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

slack_per_episode = []
term_slack_per_episode = []
x_preds_per_episode = []
for episode in range(num_episodes):
    vehicle = get_vehicle()
    x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds)
    value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
    controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
    slack_per_episode.append(slacks)
    term_slack_per_episode.append(terminal_slacks)
    x_preds_per_episode.append(x_preds)

filename = "lti_lmpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, None, i=save_number)

#### ------------------------------------------------------ LBLMPC ---------------------------------------####
print("LTV TUBE LMPC")
#### ------------------------------------------------------ LBLMPC ---------------------------------------####
## Minimal Invariant Cannot Be Computed
alpha = 3
M = polytope.box2poly([[-db * alpha, db * alpha]] * n_states)
minimal_invariant = (M.A, M.b)

controller = controllers.LTV_Tube_LMPC(A, B, C,
                                    controller_horizon, 
                                    Q, R, state_reference, 
                                    state_constraints, input_constraints, minimal_invariant)

for x_init_traj, u_init_traj, value_function in zip(x_init_trajs, u_init_trajs, init_value_functions):
    controller.add_trajectory(x_init_traj[:-1,:].T, u_init_traj.T, value_function)

model = system_id.LocalLinearModel(n_states, n_inputs, h, lamb, n_sysid_pts=n_sysid_pts, n_sysid_it=n_sysid_it)
for x_init_traj, u_init_traj in zip(x_init_trajs, u_init_trajs):
    model.add_trajectory(x_init_traj, u_init_traj)

slack_per_episode = []
term_slack_per_episode = []
model_covs_per_episode = []
x_preds_per_episode = []
for episode in range(num_episodes):
    vehicle = get_vehicle()
    
    x_traj, u_traj, x_preds, u_preds, slacks, terminal_slacks, model_covs = sim_traj(vehicle, controller, input_bounds, 
                                                                                     model=model, model_callback=model_callback)
    
    value_function = control_utils.compute_traj_cost(x_traj[:-1,:].T, u_traj.T, stage_cost)
    controller.add_trajectory(x_traj[:-1,:].T, u_traj.T, value_function)
    model.add_trajectory(x_traj, u_traj)
    
    As, Bs, Cs, covs, errors = model.regress_models(x_traj[:controller.N,:], u_traj[:controller.N,:])
    controller.set_models(As, Bs, Cs)
    
    slack_per_episode.append(slacks)
    term_slack_per_episode.append(terminal_slacks)
    model_covs_per_episode.append(model_covs)
    x_preds_per_episode.append(x_preds)

filename = "tube_ltv_lmpc"
if save_data:
    experiment_utils.save_result(save_dir, filename, controller.traj_list, controller.input_traj_list, x_preds_per_episode, controller.value_func_list, model_covs_per_episode,i=save_number)

print("FINISHED EXPERIMENT")
