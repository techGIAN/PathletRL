import json
import numpy as np
import random as rnd
import time
from datetime import dt
import os


def read_json_dict(json_file):

    with open(json_file, 'r') as f:
        new_dict = json.load(f)
    
    return new_dict

def compute_avg_attributes(environment, policy, num_episodes=10):
    '''
        Comptues the average [attribute] of a policy on the environment over n episodes
        where [attribute] = {size of pathlet set, phi, (un)weighted representatbilities, traj_losses (abs and %), rewards }
        Params:
        -------
        environment: The PathletGraphEnvironment
        policy: The policy of the reinforcement learning
        num_episodes: Number of episodes
        Return:
        -------
        The average of the attributes
    '''
    sizes_of_S, phis, reprs, w_reprs, traj_losses, traj_losses_percents, rewards = [], [], [], [], [], [], []
    total_return = 0.0
    best_reward = 0
    best_S = None

    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        i = 1
        n_epi = 0
        print('About to start...')
        steps_taken = 0
        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward

            if time_step.reward[0] == 0.0:
                return 0,0,0,0,0,0,0,0,{}
            print(f'{i} - Still going with step reward {time_step.reward[0]}, reward so far = {episode_return[0]}')
            
            prev_rew = time_step.reward[0]
            first_step = False
            i += 1
            steps_taken += 1
        
        total_return += episode_return
        n_epi += 1
        print(f'Done with final episode return {episode_return[0]}, total_return so far = {total_return}, ave_return = {total_return/n_epi}')
        attributes = environment.pyenv.envs[0].episode_attributes()
        size_S, phi, repre, w_repre, L_traj, L_traj_loss, R, set_S = attributes.values()
        if R > best_reward:
            best_reward = R
            best_S = set_S
        if best_S is None:
            best_S = set_S
        
        sizes_of_S.append(size_S)
        phis.append(phi)
        reprs.append(repre)
        w_reprs.append(w_repre)
        traj_losses.append(L_traj)
        traj_losses_percents.append(L_traj_loss)
        rewards.append(R)
    
    avg_return = total_return / num_episodes
    avg_return = avg_return.numpy()[0]

    print(f'Finished computing attributes: {datetime.now()}')

    return np.mean(sizes_of_S), \
            np.mean(phis), \
            np.mean(reprs), \
            np.mean(w_reprs), \
            np.mean(traj_losses), \
            np.mean(traj_losses_percents), \
            np.mean(rewards), \
            avg_return, best_S

def collect_step(environment, policy, buffer):
    '''
        Helper method for the collect_step
    '''
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, 
                                      action_step, 
                                      next_time_step)
    
    buffer.add_batch(traj)

def run_rl(tf_train_env,
            tf_eval_env,
            model,
            agent,
            starting_avg_reward,
            weighted,
            out_filename,
            n_episodes=5,
            n_iters=100,
            loss_step=1,
            perf_step=10,
            max_len_buffer=100000,
            replay_buffer=None,
            collect_steps_per_iteration=1,
            batch_size=64,
            dataset=None,
            rep_buff_dset_steps=2,
            rep_buff_dset_prefetch=3):

    '''
        Helper method for running the RL method
        Params:
        -------
        tf_env: The tensorflow version of the environment
        agent: The tf-agents agent
        starting_avg_reward: The initial reward
        weighted: If weighted or not
        out_filename: Filename
        n_episodes: Number of episodes to run on
        n_iters: Number of iterations for running the model
        loss_step: Report the loss every loss_steps
        perf_step: Report the attributes/performance every perf_steps
        max_len_buffer: The max length of the replay buffer
        replay_buffer: If none (during train), then set it up. Else, use the one we have
        collect_steps_per_iteration: Number of collect steps per iteration
        batch_size: The batch size for defining the dataset
        dataset: The dataset. If none (during train), then set it up. Else, use the one we have
        num_parallel_calls: Number of parallel calls for the dataset
        rep_buff_dset_steps: The number of steps for the replay buffer for the dataset
        rep_buff_dset_prefetch: The number of prefetch for the replay buffer for the dataset
        Returns:
        --------
        reward
        loss
        dataset
    '''

    s = 'Step, Reward, S, phi, L_traj_percRNDent, representability, loss\n'

    start_time = time.perf_counter()
    print(f'Time started: {datetime.now()}')

    avg_reward_values, losses, ave_returns = [starting_avg_reward], [0], []

    print('Setting up replay buffer...')

    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
                                    data_spec=agent.collect_data_spec,                                                                
                                    batch_size=tf_train_env.batch_size,                                                              
                                    max_length=max_len_buffer)

    dataset = replay_buffer.as_dataset(num_parallel_calls=4, 
                                    sample_batch_size=batch_size, 
                                    num_steps=rep_buff_dset_steps).prefetch(rep_buff_dset_prefetch)
    
    print(f'Starting TF environment at time: {datetime.now()}')

    iterator = iter(dataset)
    print('Starting tf environment...')
    tf_train_env.reset()

    print(f'Collecting step at time: {datetime.now()}')
    print('Now collecting some steps...')
    for _ in range(batch_size):
        collect_step(tf_train_env, agent.policy, replay_buffer)

    print(f'Running at time: {datetime.now()}')
    print('Running...')
    best_S, best_ave_return = None, 0

    for i in range(n_iters):
        
        print(f'Collecting collect_policy at time: {datetime.now()}')
        for _ in range(collect_steps_per_iteration):
            collect_step(tf_train_env, agent.collect_policy, replay_buffer)
        
        print(f'Finished collecting collect_policy and now loss at time: {datetime.now()}')
        experience, unused_info = next(iterator)
        if model == 'DQN-':
            my_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()

        if model == 'RND-' and i % loss_step == 0:
            print(f'Step {i}: ')
        elif model == 'DQN-' and step % loss_step == 0:
            print(f'Step {step}: loss = {my_loss}')
            losses.append(my_loss)

        print('Evaluating...')

        if model == 'DQN-' and step % perf_step == 0:

            set_S = {}
            while len(set_S) == 0:
                ave_S, ave_phi, ave_repr, ave_w_repr, ave_L_traj, ave_L_traj_percent, ave_R, ave_return, set_S = compute_avg_attributes(tf_eval_env, agent.policy, n_episodes)
            avg_reward_values.append(ave_R)
            ave_returns.append(ave_return)
        
            if weighted:
                ave_repr = ave_w_repr

            if best_ave_return < ave_return:
                best_ave_return = ave_return
                best_S = set_S

            my_loss_num = my_loss.numpy()
            print(f'Step {step} -- AVE reward: {ave_R}, |S| = {ave_S}, phi = {ave_phi}, L_traj_percent = {ave_L_traj_percent}, representability: {ave_repr}, loss = {my_loss_num}, ave_return = {ave_return}, time = {datetime.now()}')
            s += ','.join([str(step), str(ave_R), str(ave_S), str(ave_phi), str(ave_L_traj_percent), str(ave_repr), str(my_loss_num)]) + '\n'
        
        elif model == 'RND-' and i % perf_step == 0:

            set_S = {}
            while len(set_S) == 0:
                ave_S, ave_phi, ave_repr, ave_w_repr, ave_L_traj, ave_L_traj_percent, ave_R, ave_return, set_S = compute_avg_attributes(tf_eval_env, agent.policy, n_episodes)
            avg_reward_values.append(ave_R)
            ave_returns.append(ave_return)
        
            if weighted:
                ave_repr = ave_w_repr

            if best_ave_return < ave_return:
                best_ave_return = ave_return
                best_S = set_S

            my_loss_num = 'N/A'
            print(f'Step {i} -- AVE reward: {ave_R}, |S| = {ave_S}, phi = {ave_phi}, L_traj_percent = {ave_L_traj_percent}, representability: {ave_repr}, loss = {my_loss_num}, ave_return = {ave_return}, time = {datetime.now()}')
            s += ','.join([str(i), str(ave_R), str(ave_S), str(ave_phi), str(ave_L_traj_percent), str(ave_repr), str(my_loss_num)]) + '\n'

            
    tf_train_env.close()
    tf_eval_env.close()
    print('Running complete!')

    end_time = time.perf_counter()
    print(datetime.now())

    execution_time = end_time - start_time
    print(f"The execution time is: {execution_time}")

    s += 'Execution time: ' + str(execution_time)
    f = open(out_filename + '.txt', 'w')
    f.write(s)
    f.close()

    return avg_reward_values, ave_returns, losses, dataset, replay_buffer

def read_datasets(data_name):
    reduced = ''
    if data_name == 'rome':
        reduced = 'reduced_'
    pathlets_json_files = ['./data/' + data_name + '/' + reduced + 'pathlet' + txt +  '_dict.json' for txt in ['', '_rev']]
    my_edge_dict = read_json_dict(pathlets_json_files[0])
    my_edge_rev_dict = read_json_dict(pathlet_json_files[1])

    traj_json_files = ['./data/' + data_name + '/' + reduced + 'traj_' + txt +  '_dict.json' for txt in ['pathlets', 'edge']]
    my_traj_pathlets_dict = read_json_dict(traj_json_files[0])
    my_traj_edge_dict = read_json_dict(traj_json_files[1])

    return my_edge_dict, my_edge_rev_dict, my_traj_pathlets_dict, my_traj_edge_dict