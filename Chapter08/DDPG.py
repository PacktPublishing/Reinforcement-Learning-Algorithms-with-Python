import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
from collections import deque
import time

current_milli_time = lambda: int(round(time.time() * 1000))

def mlp(x, hidden_layers, output_layer, activation=tf.nn.relu, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

def deterministic_actor_critic(x, a, hidden_sizes, act_dim, max_act):
    '''
    Deterministic Actor-Critic
    '''
    # Actor
    with tf.variable_scope('p_mlp'):
        p_means = max_act * mlp(x, hidden_sizes, act_dim, last_activation=tf.tanh) 
    
    # Critic with as input the deterministic action of the actor
    with tf.variable_scope('q_mlp'):
        q_d = mlp(tf.concat([x,p_means], axis=-1), hidden_sizes, 1, last_activation=None) 
    
    # Critic with as input an arbirtary action
    with tf.variable_scope('q_mlp', reuse=True): # Use the weights of the mlp just defined
        q_a = mlp(tf.concat([x,a], axis=-1), hidden_sizes, 1, last_activation=None)

    return p_means, tf.squeeze(q_d), tf.squeeze(q_a)

class ExperiencedBuffer():
    '''
    Experienced buffer
    '''
    def __init__(self, buffer_size):
        # Contains up to 'buffer_size' experience
        self.obs_buf = deque(maxlen=buffer_size)
        self.rew_buf = deque(maxlen=buffer_size)
        self.act_buf = deque(maxlen=buffer_size)
        self.obs2_buf = deque(maxlen=buffer_size)
        self.done_buf = deque(maxlen=buffer_size)


    def add(self, obs, rew, act, obs2, done):
        '''
        Add a new transition to the buffers
        '''
        self.obs_buf.append(obs)
        self.rew_buf.append(rew)
        self.act_buf.append(act)
        self.obs2_buf.append(obs2)
        self.done_buf.append(done)
        

    def sample_minibatch(self, batch_size):
        '''
        Sample a mini-batch of size 'batch_size'
        '''
        mb_indices = np.random.randint(len(self.obs_buf), size=batch_size)

        mb_obs = [self.obs_buf[i] for i in mb_indices]
        mb_rew = [self.rew_buf[i] for i in mb_indices]
        mb_act = [self.act_buf[i] for i in mb_indices]
        mb_obs2 = [self.obs2_buf[i] for i in mb_indices]
        mb_done = [self.done_buf[i] for i in mb_indices]

        return mb_obs, mb_rew, mb_act, mb_obs2, mb_done

    def __len__(self):
        return len(self.obs_buf)

def test_agent(env_test, agent_op, num_games=10):
    '''
    Test an agent 'agent_op', 'num_games' times
    Return mean and std
    '''
    games_r = []
    for _ in range(num_games):
        d = False
        game_r = 0
        o = env_test.reset()

        while not d:
            a_s = agent_op(o)
            o, r, d, _ = env_test.step(a_s)
            game_r += r

        games_r.append(game_r)
    return np.mean(games_r), np.std(games_r)



def DDPG(env_name, hidden_sizes=[32], ac_lr=1e-2, cr_lr=1e-2, num_epochs=2000, buffer_size=5000, discount=0.99, render_cycle=100, mean_summaries_steps=1000, 
        batch_size=128, min_buffer_size=5000, tau=0.005):

    # Create an environment for training
    env = gym.make(env_name)
    # Create an environment for testing the actor
    env_test = gym.make(env_name)

    tf.reset_default_graph()

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape
    print('-- Observation space:', obs_dim, ' Action space:', act_dim, '--')

    # Create some placeholders
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='obs')
    act_ph = tf.placeholder(shape=(None, act_dim[0]), dtype=tf.float32, name='act')
    y_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='y')

    # Create an online deterministic actor-critic 
    with tf.variable_scope('online'):
        p_onl, qd_onl, qa_onl = deterministic_actor_critic(obs_ph, act_ph, hidden_sizes, act_dim[0], np.max(env.action_space.high))
    # and a target one
    with tf.variable_scope('target'):
        _, qd_tar, _ = deterministic_actor_critic(obs_ph, act_ph, hidden_sizes, act_dim[0], np.max(env.action_space.high))

    def variables_in_scope(scope):
        '''
        Retrieve all the variables in the scope 'scope'
        '''
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope)

    # Copy all the online variables to the target networks i.e. target = online
    # Needed only at the beginning
    init_target = [target_var.assign(online_var) for target_var, online_var in zip(variables_in_scope('target'), variables_in_scope('online'))]
    init_target_op = tf.group(*init_target)

    # Soft update
    update_target = [target_var.assign(tau*online_var + (1-tau)*target_var) for target_var, online_var in zip(variables_in_scope('target'), variables_in_scope('online'))]
    update_target_op = tf.group(*update_target)

    # Critic loss (MSE)
    q_loss = tf.reduce_mean((qa_onl - y_ph)**2) 
    # Actor loss
    p_loss = -tf.reduce_mean(qd_onl)

    # Optimize the critic
    q_opt = tf.train.AdamOptimizer(cr_lr).minimize(q_loss)
    # Optimize the actor
    p_opt = tf.train.AdamOptimizer(ac_lr).minimize(p_loss, var_list=variables_in_scope('online/p_mlp'))


    def agent_op(o):
        a = np.squeeze(sess.run(p_onl, feed_dict={obs_ph:[o]}))
        return np.clip(a, env.action_space.low, env.action_space.high)

    def agent_noisy_op(o, scale):
        action = agent_op(o)
        noisy_action = action + np.random.normal(loc=0.0, scale=scale, size=action.shape)
        return np.clip(noisy_action, env.action_space.low, env.action_space.high)


    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, int(now.second))
    print('Time:', clock_time)


    # Set TensorBoard
    tf.summary.scalar('loss/q', q_loss)
    tf.summary.scalar('loss/p', p_loss)
    scalar_summary = tf.summary.merge_all()

    hyp_str = '-aclr_'+str(ac_lr)+'-crlr_'+str(cr_lr)+'-tau_'+str(tau)

    file_writer = tf.summary.FileWriter('log_dir/'+env_name+'/DDPG_'+clock_time+'_'+hyp_str, tf.get_default_graph())

    # Create a session and initialize the variables
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    sess.run(init_target_op)
    
    # Some useful variables..
    render_the_game = False
    step_count = 0
    last_q_update_loss = []
    last_p_update_loss = []
    ep_time = current_milli_time()
    batch_rew = []

    # Reset the environment
    obs = env.reset()
    # Initialize the buffer
    buffer = ExperiencedBuffer(buffer_size)


    for ep in range(num_epochs):
        g_rew = 0
        done = False

        while not done:
            # If not gathered enough experience yet, act randomly
            if len(buffer) < min_buffer_size:
                act = env.action_space.sample()
            else:
                act = agent_noisy_op(obs, 0.1)

            # Take a step in the environment
            obs2, rew, done, _ = env.step(act)

            if render_the_game:
                env.render()

            # Add the transition in the buffer
            buffer.add(obs.copy(), rew, act, obs2.copy(), done)

            obs = obs2
            g_rew += rew
            step_count += 1

            if len(buffer) > min_buffer_size:
                # sample a mini batch from the buffer
                mb_obs, mb_rew, mb_act, mb_obs2, mb_done = buffer.sample_minibatch(batch_size)

                # Compute the target values
                q_target_mb = sess.run(qd_tar, feed_dict={obs_ph:mb_obs2})
                y_r = np.array(mb_rew) + discount*(1-np.array(mb_done))*q_target_mb

                # optimize the critic
                train_summary, _, q_train_loss = sess.run([scalar_summary, q_opt, q_loss], feed_dict={obs_ph:mb_obs, y_ph:y_r, act_ph: mb_act})
                
                # optimize the actor
                _, p_train_loss = sess.run([p_opt, p_loss], feed_dict={obs_ph:mb_obs})

                # summaries..
                file_writer.add_summary(train_summary, step_count)
                last_q_update_loss.append(q_train_loss)
                last_p_update_loss.append(p_train_loss)

                # Soft update of the target networks
                sess.run(update_target_op)

                # some 'mean' summaries to plot more smooth functions
                if step_count % mean_summaries_steps == 0:
                    summary = tf.Summary()
                    summary.value.add(tag='loss/mean_q', simple_value=np.mean(last_q_update_loss))
                    summary.value.add(tag='loss/mean_p', simple_value=np.mean(last_p_update_loss))
                    file_writer.add_summary(summary, step_count)
                    file_writer.flush()

                    last_q_update_loss = []
                    last_p_update_loss = []


            if done:
                obs = env.reset()
                batch_rew.append(g_rew)
                g_rew, render_the_game = 0, False

        # Test the actor every 10 epochs
        if ep % 10 == 0:
            test_mn_rw, test_std_rw = test_agent(env_test, agent_op)

            summary = tf.Summary()
            summary.value.add(tag='test/reward', simple_value=test_mn_rw)
            file_writer.add_summary(summary, step_count)
            file_writer.flush()

            ep_sec_time = int((current_milli_time()-ep_time) / 1000)
            print('Ep:%4d Rew:%4.2f -- Step:%5d -- Test:%4.2f %4.2f -- Time:%d' %  (ep,np.mean(batch_rew), step_count, test_mn_rw, test_std_rw, ep_sec_time))

            ep_time = current_milli_time()
            batch_rew = []
                
        if ep % render_cycle == 0:
            render_the_game = True

    # close everything
    file_writer.close()
    env.close()
    env_test.close()


if __name__ == '__main__':
    DDPG('BipedalWalker-v2', hidden_sizes=[64,64], ac_lr=3e-4, cr_lr=4e-4, buffer_size=200000, mean_summaries_steps=100, batch_size=64, 
        min_buffer_size=10000, tau=0.003)
    
