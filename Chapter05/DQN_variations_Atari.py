import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
from collections import deque
import time
import sys

from atari_wrappers import make_env


gym.logger.set_level(40)

current_milli_time = lambda: int(round(time.time() * 1000))


def cnn(x):
    '''
    Convolutional neural network
    '''
    x = tf.layers.conv2d(x, filters=16, kernel_size=8, strides=4, padding='valid', activation='relu') 
    x = tf.layers.conv2d(x, filters=32, kernel_size=4, strides=2, padding='valid', activation='relu') 
    return tf.layers.conv2d(x, filters=32, kernel_size=3, strides=1, padding='valid', activation='relu') 
    

def fnn(x, hidden_layers, output_layer, activation=tf.nn.relu, last_activation=None):
    '''
    Feed-forward neural network
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

def qnet(x, hidden_layers, output_size, fnn_activation=tf.nn.relu, last_activation=None):
    '''
    Deep Q network: CNN followed by FNN
    '''
    x = cnn(x)
    x = tf.layers.flatten(x)

    return fnn(x, hidden_layers, output_size, fnn_activation, last_activation)

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

def test_agent(env_test, agent_op, num_games=20):
    '''
    Test an agent
    '''
    games_r = []

    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            # Use an eps-greedy policy with eps=0.05 (to add stochasticity to the policy)
            # Needed because Atari envs are deterministic
            # If you would use a greedy policy, the results will be always the same
            a = eps_greedy(np.squeeze(agent_op(o)), eps=0.05)
            o, r, d, _ = env_test.step(a)

            game_r += r

        games_r.append(game_r)

    return games_r

def scale_frames(frames):
    '''
    Scale the frame with number between 0 and 1
    '''
    return np.array(frames, dtype=np.float32) / 255.0


def dueling_qnet(x, hidden_layers, output_size, fnn_activation=tf.nn.relu, last_activation=None):
    '''
    Dueling neural network
    '''
    x = cnn(x)
    x = tf.layers.flatten(x)

    qf = fnn(x, hidden_layers, 1, fnn_activation, last_activation)
    aaqf = fnn(x, hidden_layers, output_size, fnn_activation, last_activation)

    return qf + aaqf - tf.reduce_mean(aaqf)

def double_q_target_values(mini_batch_rw, mini_batch_done, target_qv, online_qv, discounted_value):   ## IS THE NAME CORRECT???
    '''
    Calculate the target value y following the double Q-learning update
    '''
    argmax_online_qv = np.argmax(online_qv, axis=1)
    
    # if episode terminate, y take value r
    # otherwise, q-learning step
    
    ys = []
    assert len(mini_batch_rw) == len(mini_batch_done) == len(target_qv) == len(argmax_online_qv)
    for r, d, t_av, arg_a in zip(mini_batch_rw, mini_batch_done, target_qv, argmax_online_qv):
        if d:
            ys.append(r)
        else:
            q_value = r + discounted_value * t_av[arg_a]
            ys.append(q_value)
    
    assert len(ys) == len(mini_batch_rw)

    return ys

class MultiStepExperienceBuffer():
    '''
    Experience Replay Buffer for multi-step learning
    '''
    def __init__(self, buffer_size, n_step, gamma):
        self.obs_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)

        self.n_obs_buf = deque(maxlen=buffer_size)
        self.n_done_buf = deque(maxlen=buffer_size)
        self.n_rew_buf = deque(maxlen=buffer_size)

        self.n_step = n_step
        self.last_rews = deque(maxlen=self.n_step+1)
        self.gamma = gamma


    def add(self, obs, rew, act, obs2, done):
        self.obs_buf.append(obs)
        self.act_buf.append(act)
        # the following buffers will be updated in the next n_step steps
        # their values are not known, yet
        self.n_obs_buf.append(None)
        self.n_rew_buf.append(None)
        self.n_done_buf.append(None)

        self.last_rews.append(rew)

        ln = len(self.obs_buf)
        len_rews = len(self.last_rews)

        # Update the indices of the buffer that are n_steps old
        if done:
            # In case it's the last step, update up to the n_steps indices fo the buffer
            # it cannot update more than len(last_rews), otherwise will update the previous traj
            for i in range(len_rews):
                self.n_obs_buf[ln-(len_rews-i-1)-1] = obs2
                self.n_done_buf[ln-(len_rews-i-1)-1] = done
                rgt = np.sum([(self.gamma**k)*r for k,r in enumerate(np.array(self.last_rews)[i:len_rews])])
                self.n_rew_buf[ln-(len_rews-i-1)-1] = rgt

            # reset the reward deque
            self.last_rews = deque(maxlen=self.n_step+1)
        else:
            # Update the elements of the buffer that has been added n_step steps ago
            # Add only if the multi-step values are updated
            if len(self.last_rews) >= (self.n_step+1):
                self.n_obs_buf[ln-self.n_step-1] = obs2
                self.n_done_buf[ln-self.n_step-1] = done
                rgt = np.sum([(self.gamma**k)*r for k,r in enumerate(np.array(self.last_rews)[:len_rews])])
                self.n_rew_buf[ln-self.n_step-1] = rgt
        

    def sample_minibatch(self, batch_size):
        # Sample a minibatch of size batch_size
        # Note: the samples should be at least of n_step steps ago
        mb_indices = np.random.randint(len(self.obs_buf)-self.n_step, size=batch_size)

        mb_obs = scale_frames([self.obs_buf[i] for i in mb_indices])
        mb_rew = [self.n_rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = scale_frames([self.n_obs_buf[i] for i in mb_indices])
        mb_done = [self.n_done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)

def DQN_with_variations(env_name, extensions_hyp, hidden_sizes=[32], lr=1e-2, num_epochs=2000, buffer_size=100000, discount=0.99, render_cycle=100, update_target_net=1000, 
        batch_size=64, update_freq=4, frames_num=2, min_buffer_size=5000, test_frequency=20, start_explor=1, end_explor=0.1, explor_steps=100000):

    # Create the environment both for train and test
    env = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    env_test = make_env(env_name, frames_num=frames_num, skip_frames=True, noop_num=20)
    # Add a monitor to the test env to store the videos
    env_test = gym.wrappers.Monitor(env_test, "VIDEOS/TEST_VIDEOS"+env_name+str(current_milli_time()),force=True, video_callable=lambda x: x%20==0)

    tf.reset_default_graph()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 

    # Create all the placeholders
    obs_ph = tf.placeholder(shape=(None, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=tf.float32, name='obs')
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='act')
    y_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')

    # Create the target network
    with tf.variable_scope('target_network'):
        if extensions_hyp['dueling']:
            target_qv = dueling_qnet(obs_ph, hidden_sizes, act_dim)
        else:
            target_qv = qnet(obs_ph, hidden_sizes, act_dim)
    target_vars = tf.trainable_variables()

    # Create the online network (i.e. the behavior policy)
    with tf.variable_scope('online_network'):
        if extensions_hyp['dueling']:
            online_qv = dueling_qnet(obs_ph, hidden_sizes, act_dim)
        else:
            online_qv = qnet(obs_ph, hidden_sizes, act_dim)
    train_vars = tf.trainable_variables()

    # Update the target network by assigning to it the variables of the online network
    # Note that the target network and the online network have the same exact architecture
    update_target = [train_vars[i].assign(train_vars[i+len(target_vars)]) for i in range(len(train_vars) - len(target_vars))]
    update_target_op = tf.group(*update_target)

    # One hot encoding of the action
    act_onehot = tf.one_hot(act_ph, depth=act_dim)
    # We are interested only in the Q-values of those actions
    q_values = tf.reduce_sum(act_onehot * online_qv, axis=1)
    
    # MSE loss function
    v_loss = tf.reduce_mean((y_ph - q_values)**2)
    # Adam optimize that minimize the loss v_loss
    v_opt = tf.train.AdamOptimizer(lr).minimize(v_loss)

    def agent_op(o):
        '''
        Forward pass to obtain the Q-values from the online network of a single observation
        '''
        # Scale the frames
        o = scale_frames(o)
        return sess.run(online_qv, feed_dict={obs_ph:[o]})

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)

    mr_v = tf.Variable(0.0)
    ml_v = tf.Variable(0.0)


    # TensorBoard summaries
    tf.summary.scalar('v_loss', v_loss)
    tf.summary.scalar('Q-value', tf.reduce_mean(q_values))
    tf.summary.histogram('Q-values', q_values)

    scalar_summary = tf.summary.merge_all()
    reward_summary = tf.summary.scalar('test_rew', mr_v)
    mean_loss_summary = tf.summary.scalar('mean_loss', ml_v)

    LOG_DIR = 'log_dir/'+env_name
    hyp_str = "-lr_{}-upTN_{}-upF_{}-frms_{}-ddqn_{}-duel_{}-nstep_{}" \
                .format(lr, update_target_net, update_freq, frames_num, extensions_hyp['DDQN'], extensions_hyp['dueling'], extensions_hyp['multi_step'])

    # initialize the File Writer for writing TensorBoard summaries
    file_writer = tf.summary.FileWriter(LOG_DIR+'/DQN_'+clock_time+'_'+hyp_str, tf.get_default_graph())

    # open a session
    sess = tf.Session()
    # and initialize all the variables
    sess.run(tf.global_variables_initializer())
    
    render_the_game = False
    step_count = 0
    last_update_loss = []
    ep_time = current_milli_time()
    batch_rew = []
    old_step_count = 0

    obs = env.reset()

    # Initialize the experience buffer
    #buffer = ExperienceBuffer(buffer_size)
    buffer = MultiStepExperienceBuffer(buffer_size, extensions_hyp['multi_step'], discount)
    
    # Copy the online network in the target network
    sess.run(update_target_op)

    ########## EXPLORATION INITIALIZATION ######
    eps = start_explor
    eps_decay = (start_explor - end_explor) / explor_steps

    for ep in range(num_epochs):
        g_rew = 0
        done = False

        # Until the environment does not end..
        while not done:
                
            # Epsilon decay
            if eps > end_explor:
                eps -= eps_decay

            # Choose an eps-greedy action 
            act = eps_greedy(np.squeeze(agent_op(obs)), eps=eps)

            # execute the action in the environment
            obs2, rew, done, _ = env.step(act)

            # Render the game if you want to
            if render_the_game:
                env.render()

            # Add the transition to the replay buffer
            buffer.add(obs, rew, act, obs2, done)

            obs = obs2
            g_rew += rew
            step_count += 1

            ################ TRAINING ###############
            # If it's time to train the network:
            if len(buffer) > min_buffer_size and (step_count % update_freq == 0):
                
                # sample a minibatch from the buffer
                mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

                if extensions_hyp['DDQN']:
                    mb_onl_qv, mb_trg_qv = sess.run([online_qv,target_qv], feed_dict={obs_ph:mb_obs2})
                    y_r = double_q_target_values(mb_rew, mb_done, mb_trg_qv, mb_onl_qv, discount)
                else:
                    mb_trg_qv = sess.run(target_qv, feed_dict={obs_ph:mb_obs2})
                    y_r = q_target_values(mb_rew, mb_done, mb_trg_qv, discount)

                # optimize, compute the loss and return the TB summary
                train_summary, train_loss, _ = sess.run([scalar_summary, v_loss, v_opt], feed_dict={obs_ph:mb_obs, y_ph:y_r, act_ph: mb_act})

                # Add the train summary to the file_writer
                file_writer.add_summary(train_summary, step_count)
                last_update_loss.append(train_loss)

            # Every update_target_net steps, update the target network
            if (len(buffer) > min_buffer_size) and (step_count % update_target_net == 0):

                # run the session to update the target network and get the mean loss sumamry 
                _, train_summary = sess.run([update_target_op, mean_loss_summary], feed_dict={ml_v:np.mean(last_update_loss)})
                file_writer.add_summary(train_summary, step_count)
                last_update_loss = []


            # If the environment is ended, reset it and initialize the variables
            if done:
                obs = env.reset()
                batch_rew.append(g_rew)
                g_rew, render_the_game = 0, False

        # every test_frequency episodes, test the agent and write some stats in TensorBoard
        if ep % test_frequency == 0:
            # Test the agent to 10 games
            test_rw = test_agent(env_test, agent_op, num_games=10)

            # Run the test stats and add them to the file_writer
            test_summary = sess.run(reward_summary, feed_dict={mr_v: np.mean(test_rw)})
            file_writer.add_summary(test_summary, step_count)

            # Print some useful stats
            ep_sec_time = int((current_milli_time()-ep_time) / 1000)
            print('Ep:%4d Rew:%4.2f, Eps:%2.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d -- Ep_Steps:%d' %
                        (ep,np.mean(batch_rew), eps, step_count, np.mean(test_rw), np.std(test_rw), ep_sec_time, (step_count-old_step_count)/test_frequency))

            ep_time = current_milli_time()
            batch_rew = []
            old_step_count = step_count
                            
        if ep % render_cycle == 0:
            render_the_game = True

    file_writer.close()
    env.close()


if __name__ == '__main__':

    extensions_hyp={
        'DDQN':False,
        'dueling':False,
        'multi_step':1
    }
    DQN_with_variations('PongNoFrameskip-v4', extensions_hyp, hidden_sizes=[128], lr=2e-4, buffer_size=100000, update_target_net=1000, batch_size=32, 
        update_freq=2, frames_num=2, min_buffer_size=10000, render_cycle=10000)