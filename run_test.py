import sys, os
from matplotlib.lines import lineStyles
sys.path.append(os.getcwd())
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import torch
from envs.quadrotor_env import QuadrotorEnv
from rcbf_sac.sac_cbf import RCBF_SAC
import numpy as np
import torch.nn.functional as F
from torch.optim import Adam
from rcbf_sac.model import GaussianPolicy
from rcbf_sac.utils import to_tensor
from gym import spaces
import time
from rcbf_sac.diff_cbf_qp import CBFQPLayer
from rcbf_sac.cbf_qp import CascadeCBFLayer
from rcbf_sac.disturbance_estimator import DisturbanceEstimator
import argparse
from rcbf_sac.dynamics import DynamicsModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
    # Environment Args
    parser.add_argument('--env-name', default="Quadrotor", help='Options are Unicycle or 2-D Quadrotor.')
    # Comet ML
    parser.add_argument('--log_comet', action='store_true', dest='log_comet', help="Whether to log data")
    parser.add_argument('--comet_key', default='', help='Comet API key')
    parser.add_argument('--comet_workspace', default='', help='Comet workspace')
    # SAC Args
    parser.add_argument('--mode', default='train', type=str, help='support option: train/test')
    parser.add_argument('--visualize', action='store_true', dest='visualize', help='visualize env -only in available test mode')
    parser.add_argument('--output', default='output', type=str, help='')
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--eval', type=bool, default=True,
                        help='Evaluates a policy a policy every 5 episode (default: True)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for reward (default: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient(τ) (default: 0.005)')
    parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                        help='learning rate (default: 0.0003)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter α determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', type=bool, default=True, metavar='G',
                        help='Automatically adjust α (default: False)')
    parser.add_argument('--seed', type=int, default=12345, metavar='N',
                        help='random seed (default: 12345)')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--max_episodes', type=int, default=400, metavar='N',
                        help='maximum number of episodes (default: 400)')
    parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                        help='hidden size (default: 256)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--start_steps', type=int, default=5000, metavar='N',
                        help='Steps sampling random actions (default: 10000)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')
    parser.add_argument('--replay_size', type=int, default=10000000, metavar='N',
                        help='size of replay buffer (default: 10000000)')
    parser.add_argument('--cuda', action="store_true",
                        help='run on CUDA (default: False)')
    parser.add_argument('--device_num', type=int, default=0, help='Select GPU number for CUDA (default: 0)')
    parser.add_argument('--resume', default='default', type=str, help='Resuming model path for testing')
    parser.add_argument('--validate_episodes', default=5, type=int, help='how many episode to perform during validate experiment')
    parser.add_argument('--validate_steps', default=1000, type=int, help='how many steps to perform a validate experiment')
    # CBF, Dynamics, Env Args
    parser.add_argument('--no_diff_qp', action='store_false', dest='diff_qp', help='Should the agent diff through the CBF?')
    parser.add_argument('--gp_model_size', default=3000, type=int, help='gp')
    parser.add_argument('--gp_max_episodes', default=100000, type=int, help='gp max train episodes.')
    parser.add_argument('--k_d', default=3.0, type=float)
    parser.add_argument('--gamma_b', default=20, type=float)
    parser.add_argument('--l_p', default=0.03, type=float,
                        help="Look-ahead distance for unicycle dynamics output.")
    # Model Based Learning
    parser.add_argument('--model_based', action='store_true', dest='model_based', help='If selected, will use data from the model to train the RL agent.')
    parser.add_argument('--real_ratio', default=0.3, type=float, help='Portion of data obtained from real replay buffer for training.')
    parser.add_argument('--k_horizon', default=1, type=int, help='horizon of model-based rollouts')
    parser.add_argument('--rollout_batch_size', default=5, type=int, help='Size of initial state batch to rollout from.')
    # Compensator
    parser.add_argument('--comp_rate', default=0.005, type=float, help='Compensator learning rate')
    parser.add_argument('--comp_train_episodes', default=200, type=int, help='Number of initial episodes to train compensator for.')
    parser.add_argument('--comp_update_episode', default=50, type=int, help='Modulo for compensator updates')
    parser.add_argument('--use_comp', type=bool, default=False, help='Should the compensator be used.')
    # L1 estimator
    parser.add_argument('--use_L1', type=bool, default=True, help='Use L1 estimator to estimate disturbance')
    
    args = parser.parse_args()



    model_path = 'output/Quadrotor-DOB-Trained'
    env = QuadrotorEnv()
    dynamics_model = DynamicsModel(env, args)
    state, obs = env.reset()
    agent = RCBF_SAC(env.observation_space.shape[0], env.action_space, env, args)
    agent.load_weights(model_path)
    dynamics_model.load_disturbance_models(model_path)


    state, obs = env.reset()
    estimator = DisturbanceEstimator(state, env)
    state_list = []
    h_list = []
    safe_action_list = []
    T1 = []
    T2 = []
    sigma = []
    gp_est_list = []
    rew = 0.0
    sigma_hat = np.zeros(state.shape[0])

    cbf_layer = CascadeCBFLayer(env, 15)
    use_cbf = False
    error = 0.0


    start = time.time()
    for i in range(0, env.max_episode_steps):
        sigma.append(sigma_hat)
        obs = env.get_obs(state, i)
        obs_tensor = to_tensor(obs, torch.FloatTensor, 'cpu')
        # action, action_comp, action_cbf = agent.select_action(obs, dynamics_model, sigma_hat=None, evaluate=True)
        # print(action_comp)
        action, _, action_mean = agent.policy.sample(obs_tensor)
        action = action.detach().cpu().numpy()
        action_mean = action_mean.detach().cpu().numpy()
        
        if use_cbf:
            safe_action = cbf_layer.get_u_safe(action, state, sigma_hat, sigma=np.zeros((6,1)), use_L1=True, disturb=env.uncertainty)
            safe_action_list.append(safe_action)
            h_list.append(cbf_layer.hh)
            action = safe_action + action
        else:
            action = action_mean
        state_GP = dynamics_model.get_state(state)
        mean, std = dynamics_model.predict_disturbance(state_GP)
        gp_est = mean + std * state
        gp_est_list.append(gp_est)
        sigma_hat = estimator.disturbance_estimator(state, action)
        state, rewards, dones, info = env.step(action)
        rew += rewards
        state_list.append(state)
        if np.linalg.norm(env.overall_disturbance[-1]) == 0.0:
            error += 0.0
        else:
            error += np.abs((np.linalg.norm(gp_est - env.overall_disturbance[-1]))/np.linalg.norm(env.overall_disturbance[-1]))*100
        if dones:
            e = error/i
            if e > 100:
                e = 0.0
            break
    end = time.time()
    print(error/env.max_episode_steps)
    print('Time: ', end-start)


    # plot trajectory
    import matplotlib.pyplot as plt
    x_ref_traj = np.array(env.x_ref_traj)
    z_ref_traj = np.array(env.z_ref_traj)
    x_ref_v = np.array(env.dx_ref_traj)
    z_ref_v = np.array(env.dz_ref_traj)
    x_ref_traj_for_draw = np.array(env.x_ref_traj_for_draw)
    z_ref_traj_for_draw = np.array(env.z_ref_traj_for_draw)
    x_ref_v_for_draw = np.array(env.dx_ref_traj_for_draw)
    z_ref_v_for_draw = np.array(env.dz_ref_traj_for_draw)
    x_bound = np.array(env.x_bound)
    z_bound = np.array(env.z_bound)
    state_list = np.array(state_list)
    h_list = np.array(h_list)
    gp_est_list = np.array(gp_est_list)
    safe_action_list = np.array(safe_action_list)
    T1 = np.array(T1)
    T2 = np.array(T2)



    disturb = np.array(env.overall_disturbance)
    sigma = np.array(sigma)


    # fig1, (ax1, ax2) = plt.subplots(1,2)
    # ax1.plot(state_list[:,0])
    # ax1.plot(state_list[:,1])
    # ax1.plot(x_ref_traj)
    # ax1.plot(z_ref_traj)
    # ax1.legend(["x", "z", "x_ref", "z_ref"])
    # ax2.plot(state_list[:,3])
    # ax2.plot(state_list[:,4])
    # ax2.plot(x_ref_v)
    # ax2.plot(z_ref_v)
    # ax2.legend(["dx", "dz", "dx_ref", "dz_ref"])
    # plt.title('Trajectory')

    # fig = plt.figure()
    # ax1.plot(state_list[:,0], state_list[:,1], color='goldenrod')
    # ax1.plot(x_ref_traj, z_ref_traj, color='blue', linestyle='dashed')
    # ax1.plot(x_bound, z_bound, color='green')
    # ax1.scatter(state_list[0,0], state_list[0,1], color='red', marker='*')
    # ax1.scatter(state_list[-1,0], state_list[-1,1], color='red', marker='*')
    # ax1.plot(np.array([x_bound[0], x_bound[-1]]), np.array([z_bound[0], z_bound[-1]]), color='green')
    # ax1.legend(["Real Trajectory", "Ref Trajetory", "Safe Bound"])
    # ax1.xlabel('x')
    # ax1.ylabel('z')



    # fig2, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # fig2.suptitle('disturbance and sigma')
    # ax1.plot(disturb[:, 0], '#13EAC9', label='disturbance')
    # ax1.plot(sigma[:, 0], '--', label='sigma')
    # ax1.legend(['disturbance', 'sigma'])
    # ax2.plot(disturb[:, 1], '#13EAC9', label='disturbance')
    # ax2.plot(sigma[:, 1], '--', label='sigma')
    # ax2.legend(['disturbance', 'sigma'])
    # ax3.plot(disturb[:, 2], '#13EAC9', label='disturbance')
    # ax3.plot(sigma[:, 2], '--', label='sigma')
    # ax3.legend(['disturbance', 'sigma'])
    # ax4.plot(disturb[:, 3], '#13EAC9', label='disturbance')
    # ax4.plot(sigma[:, 3], '--', label='sigma')
    # ax4.legend(['disturbance', 'sigma'])
    # ax5.plot(disturb[:, 4], '#13EAC9', label='disturbance')
    # ax5.plot(sigma[:, 4], '--', label='sigma')
    # ax5.legend(['disturbance', 'sigma'])
    # ax6.plot(disturb[:, 5], '#13EAC9', label='disturbance')
    # ax6.plot(sigma[:, 5], '--', label='sigma')
    # ax6.legend(['disturbance', 'sigma'])
    
    # fig2, [[ax1, ax2, ax3], [ax4, ax5, ax6]] = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
    # fig2.suptitle('disturbance and sigma')
    # ax1.plot(disturb[:, 0], '#13EAC9', label='disturbance')
    # ax1.plot(gp_est_list[:, 0], '--', label='sigma')
    # ax1.legend(['disturbance', 'GP'])
    # ax2.plot(disturb[:, 1], '#13EAC9', label='disturbance')
    # ax2.plot(gp_est_list[:, 1], '--', label='sigma')
    # ax2.legend(['disturbance', 'GP'])
    # ax3.plot(disturb[:, 2], '#13EAC9', label='disturbance')
    # ax3.plot(gp_est_list[:, 2], '--', label='sigma')
    # ax3.legend(['disturbance', 'GP'])
    # ax4.plot(disturb[:, 3], '#13EAC9', label='disturbance')
    # ax4.plot(gp_est_list[:, 3], '--', label='sigma')
    # ax4.legend(['disturbance', 'GP'])
    # ax5.plot(disturb[:, 4], '#13EAC9', label='disturbance')
    # ax5.plot(gp_est_list[:, 4], '--', label='sigma')
    # ax5.legend(['disturbance', 'GP'])
    # ax6.plot(disturb[:, 5], '#13EAC9', label='disturbance')
    # ax6.plot(gp_est_list[:, 5], '--', label='sigma')
    # ax6.legend(['disturbance', 'GP'])

    # fig3, [ax1, ax2, ax3] = plt.subplots(nrows=1, ncols=3, figsize=(15, 10))
    # ax1.plot(h_list)
    # ax2.plot(safe_action_list[:,0])
    # ax3.plot(safe_action_list[:,1])

    fig3 = plt.figure()
    c = np.tan(state_list[:,0])
    plt.plot(state_list[:,0], state_list[:,1], color='goldenrod')
    plt.plot(x_ref_traj_for_draw, z_ref_traj_for_draw, color='blue', linestyle='dashed')
    plt.plot(x_bound, z_bound, color='green')
    plt.plot([env.x_threshold, env.x_threshold], [-env.z_threshold, env.z_threshold], color='red')
    plt.plot([-env.x_threshold, -env.x_threshold], [-env.z_threshold, env.z_threshold], color='red')
    plt.plot([-env.x_threshold, env.x_threshold], [env.z_threshold, env.z_threshold], color='red')
    plt.plot([-env.x_threshold, env.x_threshold], [-env.z_threshold, -env.z_threshold], color='red')


    # plt.scatter(state_list[0,0], state_list[0,1], color='red', marker='*')
    # plt.scatter(state_list[-1,0], state_list[-1,1], color='red', marker='*')
    # plt.legend(["Real Trajectory", "Ref Trajetory", "Safe Bound"])
    # plt.title('SAC + L1 Trained Using CBF')


    # fig4, (ax1, ax2) = plt.subplots(1,2)
    # ax1.plot(T1)
    # ax1.set_title('T1')
    # ax2.plot(T2)
    # ax2.set_title('T2')

    fig5, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6)
    ax1.plot(state_list[:,0])
    ax1.set_title('x')
    ax2.plot(state_list[:,1])
    ax2.set_title('z')
    ax3.plot(state_list[:,2])
    ax3.set_title('theta')
    ax4.plot(state_list[:,3])
    ax4.set_title('dx')
    ax5.plot(state_list[:,4])
    ax5.set_title('dz')
    ax6.plot(state_list[:,5])
    ax6.set_title('dtheta')

    fig5, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(1,6)
    ax1.plot(env.x_ref_traj)
    ax1.set_title('x_ref')
    ax2.plot(env.z_ref_traj)
    ax2.set_title('z_ref')
    ax3.plot(env.theta_ref_traj)
    ax3.set_title('theta_ref')
    ax4.plot(env.dx_ref_traj)
    ax4.set_title('dx_ref')
    ax5.plot(env.dz_ref_traj)
    ax5.set_title('dz_ref')
    ax6.plot(env.dtheta_ref_traj)
    ax6.set_title('dtheta_ref')


    plt.show()
