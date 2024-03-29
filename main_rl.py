import numpy as np
import time
from datetime import datetime
import base64
import os
import argparse

from utils.util_func import read_datasets
from utils.util_func import read_json_dict
from utils.util_func import compute_avg_attributes
from utils.util_func import collect_step
from utils.util_func import run_rl

from utils.param_setup import set_params
from utils.param_setup import get_params

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf

from tf_agents.environments import py_environment
from tf_agents.environments import tf_environment
from tf_agents.environments import tf_py_environment
from tf_agents.environments import utils
from tf_agents.specs import array_spec
from tf_agents.environments import wrappers
from tf_agents.environments import suite_gym
from tf_agents.trajectories import time_step as ts
from tf_agents.trajectories import trajectory

from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.replay_buffers import tf_uniform_replay_buffer

from tf_agents.utils import common

from tf_agents.policies import random_py_policy
from tf_agents.agents.random import random_agent
from tf_agents.specs import tensor_spec
from tf_agents.policies import q_policy

from tf_agents.drivers import dynamic_episode_driver

import warnings
warnings.filterwarnings("error")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help='dataset')
    parser.add_argument("--model_name", type=str, default="DQN", help='model_name')
    parser.add_argument("--alpha1", type=float, default="0.25", help='alpha1')
    parser.add_argument("--alpha2", type=float, default="0.25", help='alpha2')
    parser.add_argument("--alpha3", type=float, default="0.25", help='alpha3')
    parser.add_argument("--alpha4", type=float, default="0.25", help='alpha4')
    parser.add_argument("--learning_rate", default="0.001", type=float, help='learning rate')
    parser.add_argument("--n_iters", type=int, default="100", help='number of iterations')
    parser.add_argument("--collect_steps_per_iteration", type=int, help='data collection for how many steps per each iteration?')
    parser.add_argument("--n_episodes", type=int, default="5", help='number of episodes')
    parser.add_argument("--replay_buffer_size", type=int, default="100000", help='size of replay buffer')
    parser.add_argument("--batch_size", default="512", type=int, help='batch size')
    parser.add_argument("--rep_buff_dset_steps", type=int, help='The number of steps for the replay buffer for the dataset')
    parser.add_argument("--rep_buff_dset_prefetch", type=int, help='The number of prefetch for the replay buffer for the dataset')
    parser.add_argument("--weighted", default="True", type=bool, help='weighted?')
    parser.add_argument("--representability", default="True", type=bool, help='with representability?')
    parser.add_argument("--psa", default="False", type=bool, help='conduct parameter sensitivity analysis?')
    
    return parser.parse_args()

starting_time = time.perf_counter()
print(f'Starting time: {datetime.now()}')

args = parse_args()
if args.alpha1 + args.alpha2 + args.alpha3 + args.alpha4 != 1:
    print("The alpha values do not add up to 1. Please try again.")
    exit()

parameters = set_params(args.dataset,
                        args.model_name,
                        args.alpha1,
                        args.alpha2,
                        args.alpha3,
                        args.alpha4,
                        args.learning_rate,
                        args.n_iters,
                        args.collect_steps_per_iteration,
                        args.n_episodes,
                        args.replay_buffer_size, 
                        args.rep_buff_dset_steps,
                        args.rep_buff_dset_prefetch,
                        args.weighted,
                        args.representability,
                        args.psa)

my_edge_dict, my_edge_rev_dict, my_traj_pathlets_dict, my_traj_edge_dict = read_datasets(parameters['data_name'])

environment = PathletGraphEnvironment(my_edge_dict,
                                      my_edge_rev_dict,
                                      my_traj_pathlets_dict,
                                      my_traj_edge_dict,
                                      weighted=weighted,
                                      k=k_ord,
                                      traj_loss_max=traj_loss_max,
                                      rep_threshold=rep_threshold,
                                      alpha=alpha,
                                      representability=representability)

environment = wrappers.RunStats(environment)
tf_train_env = tf_py_environment.TFPyEnvironment(environment)
tf_eval_env = tf_py_environment.TFPyEnvironment(environment)

train_step_counter = tf.Variable(0)
if model == 'DQN-':
    print('Setting up Q-Networks...')
    q_net = q_network.QNetwork(tf_train_env.observation_spec(), 
                            tf_train_env.action_spec(),
                            fc_layer_params=fc_layers,
                            dropout_layer_params=dropout_layer_params
                            )

    agent = dqn_agent.DqnAgent(tf_train_env.time_step_spec(),
                            tf_train_env.action_spec(),
                            q_network=q_net,
                            optimizer=optimizer,
                            train_step_counter=train_step_counter)
else:
    agent = random_agent.RandomAgent(tf_train_env.time_step_spec(),
                        tf_train_env.action_spec(),
                        train_step_counter=train_step_counter)
print('Initializing agent...')
agent.initialize()

print('Initializing environments and returns...')
ave_S, ave_phi, ave_repr, ave_w_repr, ave_L_traj, ave_L_traj_percent, ave_R, avg_return, best_S = compute_avg_attributes(tf_eval_env, agent.policy, 1)
print(f'Finished initializing: {datetime.now()}')

print('TRAINING...')
rewards, returns, losses, dataset, replay_buffer = run_rl(tf_train_env=tf_train_env,
                                                        tf_eval_env=tf_eval_env, model=model,
                                                        agent=agent,
                                                        starting_avg_reward=ave_R,
                                                        weighted=weighted,
                                                        out_filename=out_filename,
                                                        n_episodes=n_episodes,
                                                        n_iters=n_iters,
                                                        loss_step=loss_step,
                                                        perf_step=perf_step,
                                                        max_len_buffer=max_len_buffer,
                                                        replay_buffer=None,
                                                        collect_steps_per_iteration=collect_steps_per_iteration,
                                                        batch_size=batch_size,
                                                        dataset=None,
                                                        rep_buff_dset_steps=rep_buff_dset_steps,
                                                        rep_buff_dset_prefetch=rep_buff_dset_prefetch)
print(f'Finished training: {datetime.now()}')


print(f'Average reward: {np.mean(rewards)}')
print(f'Average return: {np.mean(returns)}')
print(f'Average loss: {np.mean(losses)}')

s = 'Average reward: ' + str(np.mean(rewards)) + '\n[\n'
for num in rewards:
    s += '\t' + str(num) + '\n'
s += ']'
f = open(viz_filename+'rewards.txt', 'w')
f.write(s)
f.close()

s = 'Return: ' + str(np.mean(returns)) + '\n[\n'
for num in returns:
    s += '\t' + str(num) + '\n'
s += ']'
f = open(viz_filename+'returns.txt', 'w')
f.write(s)
f.close()

if model == 'DQN-':
    s = 'Loss: ' + str(np.mean(losses)) + '\n[\n'
    for num in losses:
        s += '\t' + str(num) + '\n'
    s += ']'
    f = open(viz_filename+'losses.txt', 'w')
    f.write(s)
    f.close()

ending_time = time.perf_counter()
print(f'Ending time: {datetime.now()}')

final_execution_time = ending_time - starting_time
print(f"The execution time is: {final_execution_time}")

tf_train_env.close()
tf_eval_env.close()