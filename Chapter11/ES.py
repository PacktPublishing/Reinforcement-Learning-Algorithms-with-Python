import numpy as np 
import tensorflow as tf
from datetime import datetime
import time
import gym

import multiprocessing as mp
import scipy.stats as ss
import contextlib
import numpy as np

@contextlib.contextmanager
def temp_seed(seed):
    state = np.random.get_state()
    np.random.seed(seed)
    try:
        yield
    finally:
        np.random.set_state(state)

def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
        
    return tf.layers.dense(x, units=output_layer, activation=last_activation)


def test_agent(env_test, agent_op, num_games=1):
    '''
    Test an agent 'agent_op', 'num_games' times
    Return mean and std
    '''
    games_r = []
    steps = 0
    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            a_s = agent_op(o)
            o, r, d, _ = env_test.step(a_s)
            game_r += r
            steps += 1

        games_r.append(game_r)
    return games_r, steps


def worker(env_name, initial_seed, hidden_sizes, lr, std_noise, indiv_per_worker, worker_name, params_queue, output_queue):

    env = gym.make(env_name)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    import tensorflow as tf

    # set an initial seed common to all the workers
    tf.random.set_random_seed(initial_seed)
    np.random.seed(initial_seed)
    

    with tf.device("/cpu:" + worker_name):
        
        obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32, name='obs_ph')
        new_weights_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='new_weights_ph')
        
        def variables_in_scope(scope):
            # get all trainable variables in 'scope'
            return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)

        with tf.variable_scope('nn_' + worker_name):
            acts = mlp(obs_ph, hidden_sizes, act_dim, tf.tanh, last_activation=tf.tanh)

        agent_variables = variables_in_scope('nn_' + worker_name)
        agent_variables_flatten = flatten_list(agent_variables)

        # Update the agent parameters with new weights new_weights_ph
        it_v1 = tf.Variable(0, trainable=False)
        update_weights = []
        for a_v in agent_variables:
            upd_rsh = tf.reshape(new_weights_ph[it_v1 : it_v1+tf.reduce_prod(a_v.shape)], shape=a_v.shape)
            update_weights.append(a_v.assign(upd_rsh))
            it_v1 += tf.reduce_prod(a_v.shape)


        # Reshape the new_weights_ph following the neural network shape
        it_v2 = tf.Variable(0, trainable=False)
        vars_grads_list = []
        for a_v in agent_variables:
            vars_grads_list.append(tf.reshape(new_weights_ph[it_v2 : it_v2+tf.reduce_prod(a_v.shape)], shape=a_v.shape))
            it_v2 += tf.reduce_prod(a_v.shape)

        # Create the optimizer
        opt = tf.train.AdamOptimizer(lr)
        # Apply the "gradients" using Adam
        apply_g = opt.apply_gradients([(g, v) for g, v in zip(vars_grads_list, agent_variables)])
        
    def agent_op(o):
        a = np.squeeze(sess.run(acts, feed_dict={obs_ph:[o]}))
        return np.clip(a, env.action_space.low, env.action_space.high)


    def evaluation_on_noise(noise):
        '''
        Evaluate the agent with the noise
        ''' 
        # Get the original weights that will be restored after the evaluation
        original_weights = sess.run(agent_variables_flatten)

        # Update the weights of the agent/individual by adding the extra noise noise*STD_NOISE
        sess.run(update_weights, feed_dict={new_weights_ph:original_weights + noise*std_noise})

        # Test the agent with the new weights
        rewards, steps = test_agent(env, agent_op)

        # Restore the original weights
        sess.run(update_weights, feed_dict={new_weights_ph:original_weights})

        return np.mean(rewards), steps

    config_proto = tf.ConfigProto(device_count={'CPU': 4}, allow_soft_placement=True)
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())


    agent_flatten_shape = sess.run(agent_variables_flatten).shape

    while True:

        for _ in range(indiv_per_worker):
            seed = np.random.randint(1e7)

            with temp_seed(seed):
                # sample, for each weight of the agent, from a normal distribution
                sampled_noise = np.random.normal(size=agent_flatten_shape)
            
            # Mirrored sampling
            pos_rew, stp1 = evaluation_on_noise(sampled_noise)
            neg_rew, stp2 = evaluation_on_noise(-sampled_noise)

            # Put the returns and seeds on the queue
            # Note that here we are just sending the seed (a scalar value), not the complete perturbation sampled_noise
            output_queue.put([[pos_rew, neg_rew], seed, stp1+stp2])

        # Get all the returns and seed from each other worker
        batch_return, batch_seed = params_queue.get()

        batch_noise = []
        for seed in batch_seed:

            # reconstruct the perturbations from the seed
            with temp_seed(seed):
                sampled_noise = np.random.normal(size=agent_flatten_shape)

            batch_noise.append(sampled_noise)
            batch_noise.append(-sampled_noise)
            

        # Compute the sthocastic gradient estimate 
        vars_grads = np.zeros(agent_flatten_shape)
        for n, r in zip(batch_noise, batch_return):
            vars_grads += n * r
        vars_grads /= len(batch_noise) * std_noise

        # run Adam optimization on the estimate gradient just computed
        sess.run(apply_g, feed_dict={new_weights_ph:-vars_grads})


def normalized_rank(rewards):
    '''
    Rank the rewards and normalize them.
    '''
    ranked = ss.rankdata(rewards)
    norm = (ranked - 1) / (len(ranked) - 1)
    norm -= 0.5
    return norm


def flatten(tensor):
    '''
    Flatten a tensor
    '''
    return tf.reshape(tensor, shape=(-1,))

def flatten_list(tensor_list):
    '''
    Flatten a list of tensors
    '''
    return tf.concat([flatten(t) for t in tensor_list], axis=0)



def ES(env_name, hidden_sizes=[8,8], number_iter=1000, num_workers=4, lr=0.01, indiv_per_worker=10, std_noise=0.01):


    initial_seed = np.random.randint(1e7)

    # Create a queue for the output values (single returns and seeds values)
    output_queue = mp.Queue(maxsize=num_workers*indiv_per_worker)
    # Create a queue for the input paramaters (batch return and batch seeds)
    params_queue = mp.Queue(maxsize=num_workers)


    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    hyp_str = '-numworkers_'+str(num_workers)+'-lr_'+str(lr)
    file_writer = tf.summary.FileWriter('log_dir/'+env_name+'/'+clock_time+'_'+hyp_str, tf.get_default_graph())
    
    processes = []
    # Create a parallel process for each worker
    for widx in range(num_workers):
        p = mp.Process(target=worker, args=(env_name, initial_seed, hidden_sizes, lr, std_noise, indiv_per_worker, str(widx), params_queue, output_queue))
        p.start()
        processes.append(p)

    tot_steps = 0
    # Iterate over all the training iterations
    for n_iter in range(number_iter):

        batch_seed = []
        batch_return = []
        
        # Wait until enough candidate individuals are evaluated
        for _ in range(num_workers*indiv_per_worker):
            p_rews, p_seed, p_steps = output_queue.get()

            batch_seed.append(p_seed)
            batch_return.extend(p_rews)
            tot_steps += p_steps

        print('Iter: {} Reward: {:.2f}'.format(n_iter, np.mean(batch_return)))

        # Let's save the population's performance
        summary = tf.Summary()
        for r in batch_return:
            summary.value.add(tag='performance', simple_value=r)
        file_writer.add_summary(summary, tot_steps)
        file_writer.flush()

        # Rank and normalize the returns
        batch_return = normalized_rank(batch_return)

        # Put on the queue all the returns and seed so that each worker can optimize the neural network
        for _ in range(num_workers):
            params_queue.put([batch_return, batch_seed])
    
    # terminate all workers
    for p in processes:
        p.terminate()


        
if __name__ == '__main__':
    ES('LunarLanderContinuous-v2', hidden_sizes=[32,32], number_iter=200, num_workers=4, lr=0.02, indiv_per_worker=12, std_noise=0.05)
