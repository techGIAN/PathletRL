# setup your desired params here
def set_params():
    run = 1
    run_val = str(run)

    data_names = ['toy', 'dt', 'rome']
    data_names = {item:item for item in data_names}
    k_orders = {'toy': 4, 'dt': 10, 'rome': 10}
    traj_loss_maxes = {'dt': 0.25, 'rome': 0.25}
    rep_thresholds = {'dt': 0.8, 'rome': 0.8}

    models = {'dqn': 'DQN-', 'rnd': 'RND-'}

    data_name = 'dt'
    k_ord = k_orders[data_name]
    model = models['dqn']


    traj_loss_max = traj_loss_maxes[data_name]
    rep_threshold = rep_thresholds[data_name]
    alpha = [1/4,1/4,1/4,1/4]
    lr = 0.001
    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=lr)
    fc_layers = (128, 64, 32)
    dropout_layer_params = (0.2, 0.2, 0.1)

    n_iters = 100
    collect_steps_per_iteration = 1
    loss_step = 10
    perf_step = 20
    n_episodes = 1

    max_len_buffer = 100000
    batch_size = 32
    rep_buff_dset_steps = 2
    rep_buff_dset_prefetch = 3

    weighted = True
    weight_value = 'weighted' if weighted else 'unweighted'

    representability = True
    rep_val = 'rep-' if representability else 'no-rep-'

    PSA = False 
    param = 'alpha1='   
    param_val = 0.1 
    PSA_name = 'psa_' + param + str(param_val) + '_' if PSA else ''
    param_pos = int(param[-2])-1
    param_static = (1-param_val)/3
    alpha = alpha if not PSA else [param_static]*4
    alpha[param_pos] = alpha[param_pos] if not PSA else param_val

    viz_filename = './visualizations/opt-' + PSA_name + rep_val + 'run' + run_val + '-' + model + weight_value + '-rl-viz-' + data_name + '-'
    out_filename = './outputs/opt-' + PSA_name + rep_val +'run-' + run_val + '-' + model + weight_value + '-rl-out-' + data_name + '-'

    return {
        'run': run,
        'run_val': run_val,
        'data_names': data_names,
        'k_orders': k_orders,
        'traj_loss_maxes': traj_loss_maxes,
        'rep_thresholds': rep_thresholds,
        'models': models,
        'data_name' : data_name,
        'k_ord' : k_ord,
        'model' : model,
        'traj_loss_max': traj_loss_max,
        'rep_threshold': rep_threshold,
        'alpha': alpha,
        'lr': lr,
        'optimizer': optimizer,
        'fc_layers': fc_layers,
        'dropout_layer_params': dropout_layer_params,
        'n_iters': n_iters,
        'collect_steps_per_iteration': collect_steps_per_iteration,
        'loss_step': loss_step,
        'perf_step': perf_step,
        'n_episodes': n_episodes,
        'max_len_buffer': max_len_buffer,
        'batch_size': batch_size,
        'rep_buff_dset_steps': rep_buff_dset_steps,
        'rep_buff_dset_prefetch': rep_buff_dset_prefetch,
        'weighted': weighted,
        'weight_value': weight_value,
        'representability': representability,
        'rep_val': rep_val,
        'PSA': PSA, 
        'param': param,   
        'param_val': param_val, 
        'PSA_name': PSA_name,
        'param_pos': param_pos,
        'param_static': param_static,
        'alpha': alpha,
        'viz_filename': viz_filename,
        'out_filename': out_filename
    }

def get_params(key_param):

    return set_params()[key_param]