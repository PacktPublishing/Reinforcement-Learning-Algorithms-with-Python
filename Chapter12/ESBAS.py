import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
from collections import deque
import time
import sys


gym.logger.set_level(40)

current_milli_time = lambda: int(round(time.time() * 1000))
    

def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
        
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

class ExperienceBuffer():
    '''
    Experience Replay Buffer
    '''
    def __init__(self, buffer_size):
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)


    def add(self, obs, rew, act, obs2, done):
        # Add a new transition to the buffers
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)
        

    def sample_minibatch(self, batch_size):
        # Sample a minibatch of size batch_size
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)

        mb_obs = [self.obs_buf[i] for i in mb_indices]
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = [self.obs2_buf[i] for i in mb_indices]
        mb_done = [self.done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)


def q_target_values(mini_batch_rw, mini_batch_done, av, discounted_value):   
    '''
    Calculate the target value y for each transition
    '''
    max_av = np.max(av, axis=1)
    
    # if episode terminate, y take value r
    # otherwise, q-learning step
    ys = []
    for r, d, av in zip(mini_batch_rw, mini_batch_done, max_av):
        if d:
            ys.append(r)
        else:
            q_step = r + discounted_value * av
            ys.append(q_step)
    
    assert len(ys) == len(mini_batch_rw)
    return ys

def greedy(action_values):
    '''
    Greedy policy
    '''
    return np.argmax(action_values)

def eps_greedy(action_values, eps=0.1):
    '''
    Eps-greedy policy
    '''
    if np.random.uniform(0,1) < eps:
        # Choose a uniform random action
        return np.random.randint(len(action_values))
    else:
        # Choose the greedy action
        return np.argmax(action_values)

def test_agent(env_test, agent_op, num_games=20, summary=None):
    '''
    Test an agent
    '''
    games_r = []

    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            a = greedy(np.squeeze(agent_op(o)))
            o, r, d, _ = env_test.step(a)

            game_r += r

        if summary is not None:
            summary.value.add(tag='test_performance', simple_value=game_r)

        games_r.append(game_r)

    return games_r


class DQN_optimization:
    def __init__(self, obs_dim, act_dim, hidden_layers, lr, discount):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.hidden_layers = hidden_layers
        self.lr = lr
        self.discount = discount

        self.__build_graph()


    def __build_graph(self):
        
        self.g = tf.Graph()
        with self.g.as_default():
            # Create all the placeholders
            self.obs_ph = tf.placeholder(shape=(None, self.obs_dim[0]), dtype=tf.float32, name='obs')
            self.act_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='act')
            self.y_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')

            # Create the target network
            with tf.variable_scope('target_network'):
                self.target_qv = mlp(self.obs_ph, self.hidden_layers, self.act_dim, tf.nn.relu, last_activation=None)
            target_vars = tf.trainable_variables()

            # Create the online network (i.e. the behavior policy)
            with tf.variable_scope('online_network'):
                self.online_qv = mlp(self.obs_ph, self.hidden_layers, self.act_dim, tf.nn.relu, last_activation=None)
            train_vars = tf.trainable_variables()

            # Update the target network by assigning to it the variables of the online network
            # Note that the target network and the online network have the same exact architecture
            update_target = [train_vars[i].assign(train_vars[i+len(target_vars)]) for i in range(len(train_vars) - len(target_vars))]
            self.update_target_op = tf.group(*update_target)

            # One hot encoding of the action
            act_onehot = tf.one_hot(self.act_ph, depth=self.act_dim)
            # We are interested only in the Q-values of those actions
            q_values = tf.reduce_sum(act_onehot * self.online_qv, axis=1)
            
            # MSE loss function
            self.v_loss = tf.reduce_mean((self.y_ph - q_values)**2)
            # Adam optimize that minimize the loss v_loss
            self.v_opt = tf.train.AdamOptimizer(self.lr).minimize(self.v_loss)

            self.__create_session()

            # Copy the online network in the target network
            self.sess.run(self.update_target_op)

    def __create_session(self):
         # open a session
        self.sess = tf.Session(graph=self.g)
        # and initialize all the variables
        self.sess.run(tf.global_variables_initializer())      
    

    def act(self, o):
        '''
        Forward pass to obtain the Q-values from the online network of a single observation
        '''
        return self.sess.run(self.online_qv, feed_dict={self.obs_ph:[o]})

    def optimize(self, mb_obs, mb_rew, mb_act, mb_obs2, mb_done):
        mb_trg_qv = self.sess.run(self.target_qv, feed_dict={self.obs_ph:mb_obs2})
        y_r = q_target_values(mb_rew, mb_done, mb_trg_qv, self.discount)

        # training step
        # optimize, compute the loss and return the TB summary
        self.sess.run(self.v_opt, feed_dict={self.obs_ph:mb_obs, self.y_ph:y_r, self.act_ph: mb_act})

    def update_target_network(self):
        # run the session to update the target network and get the mean loss sumamry 
        self.sess.run(self.update_target_op)


class UCB1:
    def __init__(self, algos, epsilon):
        self.n = 0
        self.epsilon = epsilon
        self.algos = algos

        self.nk = np.zeros(len(algos))
        self.xk = np.zeros(len(algos))

    def choose_algorithm(self):
        # take the best algorithm following UCB1
        current_best = np.argmax([self.xk[i] + np.sqrt(self.epsilon * np.log(self.n) / self.nk[i]) for i in range(len(self.algos))])
        for i in range(len(self.algos)):
            if self.nk[i] < 5:
                return np.random.randint(len(self.algos))

        return current_best

    def update(self, idx_algo, traj_return):
        # Update the mean RL return 
        self.xk[idx_algo] = (self.nk[idx_algo] * self.xk[idx_algo] + traj_return) / (self.nk[idx_algo] + 1)
        # increase the number of trajectories run
        self.nk[idx_algo] += 1
        self.n += 1


def ESBAS(env_name, hidden_sizes=[32], lr=1e-2, num_epochs=2000, buffer_size=100000, discount=0.99, render_cycle=100, update_target_net=1000, 
        batch_size=64, update_freq=4, min_buffer_size=5000, test_frequency=20, start_explor=1, end_explor=0.1, explor_steps=100000,
        xi=1):

    # reset the default graph
    tf.reset_default_graph()

    # Create the environment both for train and test
    env = gym.make(env_name)
    # Add a monitor to the test env to store the videos
    env_test = gym.wrappers.Monitor(gym.make(env_name), "VIDEOS/TEST_VIDEOS"+env_name+str(current_milli_time()),force=True, video_callable=lambda x: x%20==0)

    dqns = []
    for l in hidden_sizes:
        dqns.append(DQN_optimization(env.observation_space.shape, env.action_space.n, l, lr, discount))

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)

    LOG_DIR = 'log_dir/'+env_name
    hyp_str = "-lr_{}-upTN_{}-upF_{}-xi_{}" .format(lr, update_target_net, update_freq, xi)

    # initialize the File Writer for writing TensorBoard summaries
    file_writer = tf.summary.FileWriter(LOG_DIR+'/ESBAS_'+clock_time+'_'+hyp_str, tf.get_default_graph())

    def DQNs_update(step_counter):
        # If it's time to train the network:
        if len(buffer) > min_buffer_size and (step_counter % update_freq == 0):
        
            # sample a minibatch from the buffer
            mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

            for dqn in dqns:
                dqn.optimize(mb_obs, mb_rew, mb_act, mb_obs2, mb_done)

        # Every update_target_net steps, update the target network
        if len(buffer) > min_buffer_size and (step_counter % update_target_net == 0):

            for dqn in dqns:
                dqn.update_target_network()
    

    step_count = 0
    episode = 0
    beta = 1

    # Initialize the experience buffer
    buffer = ExperienceBuffer(buffer_size)

    obs = env.reset()

    # policy exploration initialization
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps


    for ep in range(num_epochs):

        # Policies' training
        for i in range(2**(beta-1), 2**beta):
            DQNs_update(i)

        ucb1 = UCB1(dqns, xi)
        list_bests = []
        ep_rew = []
        beta += 1

        while step_count < 2**beta:

            # Chose the best policy's algortihm that will run the next trajectory 
            best_dqn = ucb1.choose_algorithm()
            list_bests.append(best_dqn)

            summary = tf.Summary()
            summary.value.add(tag='algorithm_selected', simple_value=best_dqn)
            file_writer.add_summary(summary, step_count)
            file_writer.flush()

            g_rew = 0
            done = False
                
            while not done:
                # Epsilon decay
                if eps > end_explor:
                    eps -= eps_decay
                

                # Choose an eps-greedy action 
                act = eps_greedy(np.squeeze(dqns[best_dqn].act(obs)), eps=eps)

                # execute the action in the environment
                obs2, rew, done, _ = env.step(act)

                # Add the transition to the replay buffer
                buffer.add(obs, rew, act, obs2, done)

                obs = obs2
                g_rew += rew
                step_count += 1
            

            # Update the UCB parameters of the algortihm just used
            ucb1.update(best_dqn, g_rew)

            # The environment is ended.. reset it and initialize the variables
            obs = env.reset()
            ep_rew.append(g_rew)
            g_rew = 0
            episode += 1


            # Print some stats and test the best policy
            summary = tf.Summary()
            summary.value.add(tag='train_performance', simple_value=np.mean(ep_rew))

            if episode % 10 == 0:
                unique, counts = np.unique(list_bests, return_counts=True)
                print(dict(zip(unique, counts)))

                test_agent_results = test_agent(env_test, dqns[best_dqn].act, num_games=10, summary=summary)
                print('Epoch:%4d Episode:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f Best:%2d Last:%2d' % (ep,episode,np.mean(ep_rew), eps, step_count, np.mean(test_agent_results), best_dqn, g_rew))

            file_writer.add_summary(summary, step_count)
            file_writer.flush()


    file_writer.close()
    env.close()


if __name__ == '__main__':

    #ESBAS('Acrobot-v1', hidden_sizes=[[64, 64]], lr=4e-4, buffer_size=100000, update_target_net=100, batch_size=32, 
    #    update_freq=4, min_buffer_size=100, render_cycle=10000, explor_steps=50000, num_epochs=20000, end_explor=0.1)

    ESBAS('Acrobot-v1', hidden_sizes=[[64], [16, 16], [64, 64]], lr=4e-4, buffer_size=100000, update_target_net=100, batch_size=32, 
        update_freq=4, min_buffer_size=100, render_cycle=10000, explor_steps=50000, num_epochs=20000, end_explor=0.1,
        xi=1./4)