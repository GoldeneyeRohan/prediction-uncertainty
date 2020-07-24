import glob
import re
import numpy as np
import matplotlib.pyplot as plt
import tqdm

def get_exp_num(string):
    return int(re.findall('\d+', string)[0])

def save_result(dirstring, filename, x_trajs, u_trajs, x_pred_trajs, value_funcs, model_covs, i=None):
    if i is None:
        prev_exps = glob.glob(dirstring + filename + "_*.npz")
        if len(prev_exps) > 0:
            exp_nums = [get_exp_num(exp) for exp in prev_exps]
            i = np.max(exp_nums) + 1
        else:
            i = 1
    filestring = dirstring + filename + "_" + str(i) + ".npz"
    np.savez(filestring, x_trajs=x_trajs, u_trajs=u_trajs, x_pred_trajs=x_pred_trajs, value_funcs=value_funcs, model_covs = model_covs)

def load_results(dirstring, filename):
	exps = glob.glob(dirstring + filename + "_*.npz")
	data_dicts = [np.load(exp, allow_pickle=True) for exp in exps]
	return data_dicts

def plot_with_cov(x, color="b", alpha=0.3, linestyle="-o"):
    means = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    hi = means + 2 * std
    lo = np.maximum(means - 2 * std,0)
    plt.plot(means, color + linestyle)
    plt.fill_between(np.arange(x.shape[1]), hi, lo, color=color, alpha=alpha)

def sim_traj(vehicle, controller, input_limits, episode_length, model_callback=None):
    x_traj = [vehicle.x]
    u_traj = []
    
    x_pred_trajs = []
    u_pred_trajs = []
    
    multipliers = []

    slacks = []
    terminal_slacks = []
    
    for i in tqdm.tqdm(range(episode_length)):
        u_command = controller.solve(x_traj[-1])
        if u_command is None:
            print("controller error at iteration %d" % i)
            break
        if model_callback is not None:
            A, B, C = model_callback(vehicle, controller, episode_length)
            controller.set_models(A,B,C)
        u = np.minimum(np.maximum(input_limits[0], u_command), input_limits[1])
        x_next = vehicle.f(u)

        x_traj.append(x_next)
        u_traj.append(u)
        
        if hasattr(controller, "x_traj"):
            x_pred_trajs.append(controller.x_traj.value.T)
            u_pred_trajs.append(controller.u_traj.value.T)
            if hasattr(controller, "multipliers"):
                multipliers.append(controller.multipliers.value)
            
            slacks.append(controller.slack.value)
            terminal_slacks.append(controller.terminal_slack.value)
        
    return np.array(x_traj), np.array(u_traj), x_pred_trajs, u_pred_trajs, slacks, terminal_slacks, multipliers
