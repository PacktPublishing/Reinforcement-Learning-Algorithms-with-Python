import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
import roboschool

def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return -tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)


def gaussian_log_likelihood(ac, mean, log_std):
    '''
    Gaussian Log Likelihood 
    '''
    log_p = ((ac-mean)**2 / (tf.exp(log_std)**2+1e-9) + 2*log_std) + np.log(2*np.pi)
    return -0.5 * tf.reduce_sum(log_p, axis=-1)


def conjugate_gradient(A, b, x=None, iters=10):
    '''
    Conjugate gradient method: approximate the solution of Ax=b
    It solve Ax=b without forming the full matrix, just compute the matrix-vector product (The Fisher-vector product)
    
    NB: A is not the full matrix but is a useful matrix-vector product between the averaged Fisher information matrix and arbitrary vectors 
    Descibed in Appendix C.1 of the TRPO paper
    '''
    if x is None:
        x = np.zeros_like(b)
        
    r = A(x) - b
    p = -r
    for _ in range(iters):
        a = np.dot(r, r) / (np.dot(p, A(p))+1e-8)
        x += a*p
        r_n = r + a*A(p)
        b = np.dot(r_n, r_n) / (np.dot(r, r)+1e-8)
        p = -r_n + b*p
        r = r_n
    return x

def gaussian_DKL(mu_q, log_std_q, mu_p, log_std_p):
    '''
    Gaussian KL divergence in case of a diagonal covariance matrix
    '''
    return tf.reduce_mean(tf.reduce_sum(0.5 * (log_std_p - log_std_q + tf.exp(log_std_q - log_std_p) + (mu_q - mu_p)**2 / tf.exp(log_std_p) - 1), axis=1))


def backtracking_line_search(Dkl, delta, old_loss, p=0.8):
    '''
    Backtracking line searc. It look for a coefficient s.t. the constraint on the DKL is satisfied
    It has both to
     - improve the non-linear objective
     - satisfy the constraint

    '''
    ## Explained in Appendix C of the TRPO paper
    a = 1
    it = 0
 
    new_dkl, new_loss = Dkl(a) 
    while (new_dkl > delta) or (new_loss > old_loss):
        a *= p
        it += 1
        new_dkl, new_loss = Dkl(a)

    return a



def GAE(rews, v, v_last, gamma=0.99, lam=0.95):
    '''
    Generalized Advantage Estimation
    '''
    assert len(rews) == len(v)
    vs = np.append(v, v_last)
    d = np.array(rews) + gamma*vs[1:] - vs[:-1]
    gae_advantage = discounted_rewards(d, 0, gamma*lam)
    return gae_advantage

def discounted_rewards(rews, last_sv, gamma):
    '''
    Discounted reward to go 

    Parameters:
    ----------
    rews: list of rewards
    last_sv: value of the last state
    gamma: discount value 
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1] + gamma*last_sv
    for i in reversed(range(len(rews)-1)):
        rtg[i] = rews[i] + gamma*rtg[i+1]
    return rtg

class Buffer():
    '''
    Class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99, lam=0.95):
        self.gamma = gamma
        self.lam = lam
        self.adv = []
        self.ob = []
        self.ac = []
        self.rtg = []

    def store(self, temp_traj, last_sv):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        last_sv: value of the last state (Used to Bootstrap)
        '''
        # store only if there are temporary trajectories
        if len(temp_traj) > 0:
            self.ob.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], last_sv, self.gamma)
            self.adv.extend(GAE(temp_traj[:,1], temp_traj[:,3], last_sv, self.gamma, self.lam))
            self.rtg.extend(rtg)
            self.ac.extend(temp_traj[:,2])

    def get_batch(self):
        # standardize the advantage values
        norm_adv = (self.adv - np.mean(self.adv)) / (np.std(self.adv) + 1e-10)
        return np.array(self.ob), np.array(self.ac), np.array(norm_adv), np.array(self.rtg)

    def __len__(self):
        assert(len(self.adv) == len(self.ob) == len(self.ac) == len(self.rtg))
        return len(self.ob)

def flatten_list(tensor_list):
    '''
    Flatten a list of tensors
    '''
    return tf.concat([flatten(t) for t in tensor_list], axis=0)

def flatten(tensor):
    '''
    Flatten a tensor
    '''
    return tf.reshape(tensor, shape=(-1,))


class StructEnv(gym.Wrapper):
    '''
    Gym Wrapper to store information like number of steps and total reward of the last espisode.
    '''
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.n_obs = self.env.reset()
        self.total_rew = 0
        self.len_episode = 0

    def reset(self, **kwargs):
        self.n_obs = self.env.reset(**kwargs)
        self.total_rew = 0
        self.len_episode = 0
        return self.n_obs.copy()
        
    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.total_rew += reward
        self.len_episode += 1
        return ob, reward, done, info

    def get_episode_reward(self):
        return self.total_rew

    def get_episode_length(self):
        return self.len_episode


def TRPO(env_name, hidden_sizes=[32], cr_lr=5e-3, num_epochs=50, gamma=0.99, lam=0.95, number_envs=1, 
        critic_iter=10, steps_per_env=100, delta=0.002, algorithm='TRPO', conj_iters=10, minibatch_size=1000):
    '''
    Trust Region Policy Optimization

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_sizes: list of the number of hidden units for each layer
    cr_lr: critic learning rate
    num_epochs: number of training epochs
    gamma: discount factor
    lam: lambda parameter for computing the GAE
    number_envs: number of "parallel" synchronous environments
        # NB: it isn't distributed across multiple CPUs
    critic_iter: NUmber of SGD iterations on the critic per epoch
    steps_per_env: number of steps per environment
            # NB: the total number of steps per epoch will be: steps_per_env*number_envs
    delta: Maximum KL divergence between two policies. Scalar value
    algorithm: type of algorithm. Either 'TRPO' or 'NPO'
    conj_iters: number of conjugate gradient iterations
    minibatch_size: Batch size used to train the critic
    '''

    tf.reset_default_graph()

    # Create a few environments to collect the trajectories
    envs = [StructEnv(gym.make(env_name)) for _ in range(number_envs)]

    low_action_space = envs[0].action_space.low
    high_action_space = envs[0].action_space.high

    obs_dim = envs[0].observation_space.shape
    act_dim = envs[0].action_space.shape[0]

    # Placeholders
    act_ph = tf.placeholder(shape=(None,act_dim), dtype=tf.float32, name='act')
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='obs')
    ret_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='ret')
    adv_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='adv')
    old_p_log_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='old_p_log')
    old_mu_ph = tf.placeholder(shape=(None, act_dim), dtype=tf.float32, name='old_mu')
    old_log_std_ph = tf.placeholder(shape=(act_dim), dtype=tf.float32, name='old_log_std')
    p_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='p_ph')
    # result of the conjugate gradient algorithm
    cg_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='cg')
        
    # Neural network that represent the policy
    with tf.variable_scope('actor_nn'):
        p_means = mlp(obs_ph, hidden_sizes, act_dim, tf.tanh, last_activation=tf.tanh)
        log_std = tf.get_variable(name='log_std', initializer=np.zeros(act_dim, dtype=np.float32) - 0.5)

    # Neural network that represent the value function
    with tf.variable_scope('critic_nn'):
        s_values = mlp(obs_ph, hidden_sizes, 1, tf.tanh, last_activation=None)
        s_values = tf.squeeze(s_values)    

    # Add "noise" to the predicted mean following the Guassian distribution with standard deviation e^(log_std)
    p_noisy = p_means + tf.random_normal(tf.shape(p_means), 0, 1) * tf.exp(log_std)
    # Clip the noisy actions
    a_sampl = tf.clip_by_value(p_noisy, low_action_space, high_action_space)
    # Compute the gaussian log likelihood
    p_log = gaussian_log_likelihood(act_ph, p_means, log_std)

    # Measure the divergence
    diverg = tf.reduce_mean(tf.exp(old_p_log_ph - p_log))
    
    # ratio
    ratio_new_old = tf.exp(p_log - old_p_log_ph)
    # TRPO surrogate loss function
    p_loss = - tf.reduce_mean(ratio_new_old * adv_ph)

    # MSE loss function
    v_loss = tf.reduce_mean((ret_ph - s_values)**2)
    # Critic optimization
    v_opt = tf.train.AdamOptimizer(cr_lr).minimize(v_loss)

    def variables_in_scope(scope):
        # get all trainable variables in 'scope'
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    
    # Gather and flatten the actor parameters
    p_variables = variables_in_scope('actor_nn')
    p_var_flatten = flatten_list(p_variables)

    # Gradient of the policy loss with respect to the actor parameters
    p_grads = tf.gradients(p_loss, p_variables)
    p_grads_flatten = flatten_list(p_grads)

    ########### RESTORE ACTOR PARAMETERS ###########
    p_old_variables = tf.placeholder(shape=(None,), dtype=tf.float32, name='p_old_variables')
    # variable used as index for restoring the actor's parameters
    it_v1 = tf.Variable(0, trainable=False)
    restore_params = []

    for p_v in p_variables:
        upd_rsh = tf.reshape(p_old_variables[it_v1 : it_v1+tf.reduce_prod(p_v.shape)], shape=p_v.shape)
        restore_params.append(p_v.assign(upd_rsh)) 
        it_v1 += tf.reduce_prod(p_v.shape)

    restore_params = tf.group(*restore_params)

    # gaussian KL divergence of the two policies 
    dkl_diverg = gaussian_DKL(old_mu_ph, old_log_std_ph, p_means, log_std) 

    # Jacobian of the KL divergence (Needed for the Fisher matrix-vector product)
    dkl_diverg_grad = tf.gradients(dkl_diverg, p_variables) 

    dkl_matrix_product = tf.reduce_sum(flatten_list(dkl_diverg_grad) * p_ph)
    print('dkl_matrix_product', dkl_matrix_product.shape)
    # Fisher vector product
    # The Fisher-vector product is a way to compute the A matrix without the need of the full A
    Fx = flatten_list(tf.gradients(dkl_matrix_product, p_variables))

    ## Step length
    beta_ph = tf.placeholder(shape=(), dtype=tf.float32, name='beta')
    # NPG update
    npg_update = beta_ph * cg_ph
    
    ## alpha is found through line search
    alpha = tf.Variable(1., trainable=False)
    # TRPO update
    trpo_update = alpha * npg_update

    ####################   POLICY UPDATE  ###################
    # variable used as an index
    it_v = tf.Variable(0, trainable=False)
    p_opt = []
    # Apply the updates to the policy
    for p_v in p_variables:
        upd_rsh = tf.reshape(trpo_update[it_v : it_v+tf.reduce_prod(p_v.shape)], shape=p_v.shape)
        p_opt.append(p_v.assign_sub(upd_rsh))
        it_v += tf.reduce_prod(p_v.shape)

    p_opt = tf.group(*p_opt)
        
    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)


    # Set scalars and hisograms for TensorBoard
    tf.summary.scalar('p_loss', p_loss, collections=['train'])
    tf.summary.scalar('v_loss', v_loss, collections=['train'])
    tf.summary.scalar('p_divergence', diverg, collections=['train'])
    tf.summary.scalar('ratio_new_old',tf.reduce_mean(ratio_new_old), collections=['train'])
    tf.summary.scalar('dkl_diverg', dkl_diverg, collections=['train'])
    tf.summary.scalar('alpha', alpha, collections=['train'])
    tf.summary.scalar('beta', beta_ph, collections=['train'])
    tf.summary.scalar('p_std_mn', tf.reduce_mean(tf.exp(log_std)), collections=['train'])
    tf.summary.scalar('s_values_mn', tf.reduce_mean(s_values), collections=['train'])
    tf.summary.histogram('p_log', p_log, collections=['train'])
    tf.summary.histogram('p_means', p_means, collections=['train'])
    tf.summary.histogram('s_values', s_values, collections=['train'])
    tf.summary.histogram('adv_ph',adv_ph, collections=['train'])
    tf.summary.histogram('log_std',log_std, collections=['train'])
    scalar_summary = tf.summary.merge_all('train')

    tf.summary.scalar('old_v_loss', v_loss, collections=['pre_train'])
    tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    pre_scalar_summary = tf.summary.merge_all('pre_train')

    hyp_str = '-spe_'+str(steps_per_env)+'-envs_'+str(number_envs)+'-cr_lr'+str(cr_lr)+'-crit_it_'+str(critic_iter)+'-delta_'+str(delta)+'-conj_iters_'+str(conj_iters)
    file_writer = tf.summary.FileWriter('log_dir/'+env_name+'/'+algorithm+'_'+clock_time+'_'+hyp_str, tf.get_default_graph())
    
    # create a session
    sess = tf.Session()
    # initialize the variables
    sess.run(tf.global_variables_initializer())
    
    # variable to store the total number of steps
    step_count = 0
    
    print('Env batch size:',steps_per_env, ' Batch size:',steps_per_env*number_envs)

    for ep in range(num_epochs):
        # Create the buffer that will contain the trajectories (full or partial) 
        # run with the last policy
        buffer = Buffer(gamma, lam)
        # lists to store rewards and length of the trajectories completed
        batch_rew = []
        batch_len = []

        # Execute in serial the environment, storing temporarily the trajectories.
        for env in envs:
            temp_buf = []

            # iterate over a fixed number of steps
            for _ in range(steps_per_env):
                # run the policy
                act, val = sess.run([a_sampl, s_values], feed_dict={obs_ph:[env.n_obs]})
                act = np.squeeze(act)

                # take a step in the environment
                obs2, rew, done, _ = env.step(act)

                # add the new transition to the temporary buffer
                temp_buf.append([env.n_obs.copy(), rew, act, np.squeeze(val)])

                env.n_obs = obs2.copy()
                step_count += 1

                if done:
                    # Store the full trajectory in the buffer 
                    # (the value of the last state is 0 as the trajectory is completed)
                    buffer.store(np.array(temp_buf), 0)
                    # Empty temporary buffer
                    temp_buf = []

                    batch_rew.append(env.get_episode_reward())
                    batch_len.append(env.get_episode_length())

                    env.reset()
                    
            # Bootstrap with the estimated state value of the next state!
            lsv = sess.run(s_values, feed_dict={obs_ph:[env.n_obs]})
            buffer.store(np.array(temp_buf), np.squeeze(lsv))


        # Get the entire batch from the buffer
        # NB: all the batch is used and deleted after the optimization. This is because PPO is on-policy
        obs_batch, act_batch, adv_batch, rtg_batch = buffer.get_batch()

        # log probabilities, logits and log std of the "old" policy
        # "old" policy refer to the policy to optimize and that has been used to sample from the environment
        old_p_log, old_p_means, old_log_std = sess.run([p_log, p_means, log_std], feed_dict={obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, ret_ph:rtg_batch})
        # get also the "old" parameters
        old_actor_params = sess.run(p_var_flatten)

        # old_p_loss is later used in the line search
        # run pre_scalar_summary for a summary before the optimization
        old_p_loss, summary = sess.run([p_loss,pre_scalar_summary], feed_dict={obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, ret_ph:rtg_batch, old_p_log_ph:old_p_log})
        file_writer.add_summary(summary, step_count)

        def H_f(p):
            '''
            Run the Fisher-Vector product on 'p' to approximate the Hessian of the DKL
            '''
            return sess.run(Fx, feed_dict={old_mu_ph:old_p_means, old_log_std_ph:old_log_std, p_ph:p, obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, ret_ph:rtg_batch})

        g_f = sess.run(p_grads_flatten, feed_dict={old_mu_ph:old_p_means,obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, ret_ph:rtg_batch, old_p_log_ph:old_p_log})
        ## Compute the Conjugate Gradient so to obtain an approximation of H^(-1)*g
        # Where H in reality isn't the true Hessian of the KL divergence but an approximation of it computed via Fisher-Vector Product (F)
        conj_grad = conjugate_gradient(H_f, g_f, iters=conj_iters)

        # Compute the step length
        beta_np = np.sqrt(2*delta / np.sum(conj_grad * H_f(conj_grad)))
        
        def DKL(alpha_v):
            '''
            Compute the KL divergence.
            It optimize the function to compute the DKL. Afterwards it restore the old parameters.
            '''
            sess.run(p_opt, feed_dict={beta_ph:beta_np, alpha:alpha_v, cg_ph:conj_grad, obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, old_p_log_ph:old_p_log})
            a_res = sess.run([dkl_diverg, p_loss], feed_dict={old_mu_ph:old_p_means, old_log_std_ph:old_log_std, obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, ret_ph:rtg_batch, old_p_log_ph:old_p_log})
            sess.run(restore_params, feed_dict={p_old_variables: old_actor_params})
            return a_res

        # Actor optimization step
        # Different for TRPO or NPG
        if algorithm=='TRPO':
            # Backtracing line search to find the maximum alpha coefficient s.t. the constraint is valid
            best_alpha = backtracking_line_search(DKL, delta, old_p_loss, p=0.8)
            sess.run(p_opt, feed_dict={beta_ph:beta_np, alpha:best_alpha, cg_ph:conj_grad, obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, old_p_log_ph:old_p_log})
        elif algorithm=='NPG':
            # In case of NPG, no line search
            sess.run(p_opt, feed_dict={beta_ph:beta_np, alpha:1, cg_ph:conj_grad, obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, old_p_log_ph:old_p_log})


        lb = len(buffer)
        shuffled_batch = np.arange(lb)
        np.random.shuffle(shuffled_batch)

        # Value function optimization steps
        for _ in range(critic_iter):
            # shuffle the batch on every iteration
            np.random.shuffle(shuffled_batch)
            for idx in range(0,lb, minibatch_size):
                minib = shuffled_batch[idx:min(idx+minibatch_size,lb)]
                sess.run(v_opt, feed_dict={obs_ph:obs_batch[minib], ret_ph:rtg_batch[minib]})

        # print some statistics and run the summary for visualizing it on TB
        if len(batch_rew) > 0:
            train_summary = sess.run(scalar_summary, feed_dict={beta_ph:beta_np, obs_ph:obs_batch, act_ph:act_batch, adv_ph:adv_batch, cg_ph:conj_grad,
                                                                old_p_log_ph:old_p_log, ret_ph:rtg_batch, old_mu_ph:old_p_means, old_log_std_ph:old_log_std})
            file_writer.add_summary(train_summary, step_count)
            
            summary = tf.Summary()
            summary.value.add(tag='supplementary/performance', simple_value=np.mean(batch_rew))
            summary.value.add(tag='supplementary/len', simple_value=np.mean(batch_len))
            file_writer.add_summary(summary, step_count)
            file_writer.flush()

            print('Ep:%d Rew:%.2f -- Step:%d' % (ep, np.mean(batch_rew), step_count))

    # closing environments..
    for env in envs:
        env.close()

    file_writer.close()

if __name__ == '__main__':
    TRPO('RoboschoolWalker2d-v1', hidden_sizes=[64,64], cr_lr=2e-3, gamma=0.99, lam=0.95, num_epochs=1000, steps_per_env=6000, 
         number_envs=1, critic_iter=10, delta=0.01, algorithm='TRPO', conj_iters=10, minibatch_size=1000)
