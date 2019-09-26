import numpy as np 
import tensorflow as tf
import gym
from datetime import datetime
import time


def mlp(x, hidden_layers, output_size, activation=tf.nn.relu, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_size, activation=last_activation)

def softmax_entropy(logits):
    '''
    Softmax Entropy
    '''
    return tf.reduce_sum(tf.nn.softmax(logits, axis=-1) * tf.nn.log_softmax(logits, axis=-1), axis=-1)


def discounted_rewards(rews, gamma):
    '''
    Discounted reward to go 

    Parameters:
    ----------
    rews: list of rewards
    gamma: discount value 
    '''
    rtg = np.zeros_like(rews, dtype=np.float32)
    rtg[-1] = rews[-1]
    for i in reversed(range(len(rews)-1)):
        rtg[i] = rews[i] + gamma*rtg[i+1]
    return rtg

class Buffer():
    '''
    Buffer class to store the experience from a unique policy
    '''
    def __init__(self, gamma=0.99):
        self.gamma = gamma
        self.obs = []
        self.act = []
        self.ret = []
        self.rtg = []

    def store(self, temp_traj):
        '''
        Add temp_traj values to the buffers and compute the advantage and reward to go

        Parameters:
        -----------
        temp_traj: list where each element is a list that contains: observation, reward, action, state-value
        '''
        # store only if the temp_traj list is not empty
        if len(temp_traj) > 0:
            self.obs.extend(temp_traj[:,0])
            rtg = discounted_rewards(temp_traj[:,1], self.gamma)
            # NEW
            self.ret.extend(rtg - temp_traj[:,3])
            self.rtg.extend(rtg)
            self.act.extend(temp_traj[:,2])

    def get_batch(self):
        # MODIFIED
        return self.obs, self.act, self.ret, self.rtg

    def __len__(self):
        assert(len(self.obs) == len(self.act) == len(self.ret) == len(self.rtg))
        return len(self.obs)


def REINFORCE_baseline(env_name, hidden_sizes=[32], p_lr=5e-3, vf_lr=8e-3, gamma=0.99, steps_per_epoch=100, num_epochs=1000):
    '''
    REINFORCE with baseline Algorithm

    Parameters:
    -----------
    env_name: Name of the environment
    hidden_size: list of the number of hidden units for each layer
    p_lr: policy learning rate
    vf_lr: value function learning rate
    gamma: discount factor
    steps_per_epoch: number of steps per epoch
    num_epochs: number train epochs (Note: they aren't properly epochs)
    '''
    tf.reset_default_graph()

    env = gym.make(env_name)    
    
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.n 

    # Placeholders
    obs_ph = tf.placeholder(shape=(None, obs_dim[0]), dtype=tf.float32, name='obs')
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='act')
    ret_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='ret')
    rtg_ph = tf.placeholder(shape=(None,), dtype=tf.float32, name='rtg')

    #####################################################
    ########### COMPUTE THE PG LOSS FUNCTIONS ###########
    #####################################################

    # policy
    p_logits = mlp(obs_ph, hidden_sizes, act_dim, activation=tf.tanh)

    act_multn = tf.squeeze(tf.random.multinomial(p_logits, 1))
    actions_mask = tf.one_hot(act_ph, depth=act_dim)
    p_log = tf.reduce_sum(actions_mask * tf.nn.log_softmax(p_logits), axis=1)
    # entropy useful to study the algorithms
    entropy = -tf.reduce_mean(softmax_entropy(p_logits))
    p_loss = -tf.reduce_mean(p_log*ret_ph)

    # policy optimization
    p_opt = tf.train.AdamOptimizer(p_lr).minimize(p_loss)

    #######################################
    ###########  VALUE FUNCTION ###########
    #######################################
    
    ########### NEW ###########
    # value function
    s_values = tf.squeeze(mlp(obs_ph, hidden_sizes, 1, activation=tf.tanh))

    # MSE loss function
    v_loss = tf.reduce_mean((rtg_ph - s_values)**2)

    # value function optimization
    v_opt = tf.train.AdamOptimizer(vf_lr).minimize(v_loss)

    # Time
    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    print('Time:', clock_time)


    # Set scalars and hisograms for TensorBoard
    tf.summary.scalar('p_loss', p_loss, collections=['train'])
    tf.summary.scalar('v_loss', v_loss, collections=['train'])
    tf.summary.scalar('entropy', entropy, collections=['train'])
    tf.summary.scalar('s_values', tf.reduce_mean(s_values), collections=['train'])
    tf.summary.histogram('p_soft', tf.nn.softmax(p_logits), collections=['train'])
    tf.summary.histogram('p_log', p_log, collections=['train'])
    tf.summary.histogram('act_multn', act_multn, collections=['train'])
    tf.summary.histogram('p_logits', p_logits, collections=['train'])
    tf.summary.histogram('ret_ph', ret_ph, collections=['train'])
    tf.summary.histogram('rtg_ph', rtg_ph, collections=['train'])
    tf.summary.histogram('s_values', s_values, collections=['train'])
    train_summary = tf.summary.merge_all('train')

    tf.summary.scalar('old_v_loss', v_loss, collections=['pre_train'])
    tf.summary.scalar('old_p_loss', p_loss, collections=['pre_train'])
    pre_scalar_summary = tf.summary.merge_all('pre_train')

    hyp_str = '-steps_{}-plr_{}-vflr_{}'.format(steps_per_epoch, p_lr, vf_lr)
    file_writer = tf.summary.FileWriter('log_dir/{}/REINFORCE_basel_{}_{}'.format(env_name, clock_time, hyp_str), tf.get_default_graph())
    
    # create a session
    sess = tf.Session()
    # initialize the variables
    sess.run(tf.global_variables_initializer())

    # few variables
    step_count = 0
    train_rewards = []
    train_ep_len = []
    timer = time.time()

    # main cycle
    for ep in range(num_epochs):

        # initialize environment for the new epochs
        obs = env.reset()

        # intiaizlie buffer and other variables for the new epochs
        buffer = Buffer(gamma)
        env_buf = []
        ep_rews = []
        
        while len(buffer) < steps_per_epoch:

            # run the policy
            act, val = sess.run([act_multn, s_values], feed_dict={obs_ph:[obs]})
            # take a step in the environment
            obs2, rew, done, _ = env.step(np.squeeze(act))

            # add the new transition
            env_buf.append([obs.copy(), rew, act, np.squeeze(val)])

            obs = obs2.copy()

            step_count += 1
            ep_rews.append(rew)

            if done:
                # store the trajectory just completed
                buffer.store(np.array(env_buf))
                env_buf = []
                # store additionl information about the episode
                train_rewards.append(np.sum(ep_rews))
                train_ep_len.append(len(ep_rews))
                # reset the environment
                obs = env.reset()
                ep_rews = []

        # collect the episodes' information
        obs_batch, act_batch, ret_batch, rtg_batch = buffer.get_batch()
        
        # run pre_scalar_summary before the optimization phase
        epochs_summary = sess.run(pre_scalar_summary, feed_dict={obs_ph:obs_batch, act_ph:act_batch, ret_ph:ret_batch, rtg_ph:rtg_batch})
        file_writer.add_summary(epochs_summary, step_count)

        # Optimize the NN policy and the NN value function
        sess.run([p_opt, v_opt], feed_dict={obs_ph:obs_batch, act_ph:act_batch, ret_ph:ret_batch, rtg_ph:rtg_batch})

        # run train_summary to save the summary after the optimization
        train_summary_run = sess.run(train_summary, feed_dict={obs_ph:obs_batch, act_ph:act_batch, ret_ph:ret_batch, rtg_ph:rtg_batch})
        file_writer.add_summary(train_summary_run, step_count)

        # it's time to print some useful information
        if ep % 10 == 0:
            print('Ep:%d MnRew:%.2f MxRew:%.1f EpLen:%.1f Buffer:%d -- Step:%d -- Time:%d' % (ep, np.mean(train_rewards), np.max(train_rewards), np.mean(train_ep_len), len(buffer), step_count,time.time()-timer))

            summary = tf.Summary()
            summary.value.add(tag='supplementary/len', simple_value=np.mean(train_ep_len))
            summary.value.add(tag='supplementary/train_rew', simple_value=np.mean(train_rewards))
            file_writer.add_summary(summary, step_count)
            file_writer.flush()

            timer = time.time()
            train_rewards = []
            train_ep_len = []


    env.close()
    file_writer.close()


if __name__ == '__main__':
    REINFORCE_baseline('LunarLander-v2', hidden_sizes=[64], p_lr=8e-3, vf_lr=7e-3, gamma=0.99, steps_per_epoch=1000, num_epochs=1000)