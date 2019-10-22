import numpy as np 
import tensorflow as tf
from datetime import datetime
import time
from ple.games.flappybird import FlappyBird
from ple import PLE


def mlp(x, hidden_layers, output_layer, activation=tf.tanh, last_activation=None):
    '''
    Multi-layer perceptron
    '''
    for l in hidden_layers:
        x = tf.layers.dense(x, units=l, activation=activation)
    return tf.layers.dense(x, units=output_layer, activation=last_activation)

def flappy_to_list(fd):
    '''
    Return the state dictionary as a list
    '''
    return fd['player_y'], fd['player_vel'], fd['next_pipe_dist_to_player'], fd['next_pipe_top_y'], \
            fd['next_pipe_bottom_y'], fd['next_next_pipe_dist_to_player'], fd['next_next_pipe_top_y'], \
            fd['next_next_pipe_bottom_y']

def flappy_game_state(bol):
    '''
    Normalize the game state
    '''
    stat = flappy_to_list(bol.getGameState())
    stat = (np.array(stat, dtype=np.float32) / 300.0) - 0.5
    return stat

def no_op(env, n_act=5):
    for _ in range(n_act):
        env.act(119 if np.random.randn() < 0.5 else None)


def expert():
    '''
    Load the computational graph and pretarined weights of the expert
    '''
    graph = tf.get_default_graph()

    sess_expert = tf.Session(graph=graph)

    saver = tf.train.import_meta_graph('expert/model.ckpt.meta')
    saver.restore(sess_expert,tf.train.latest_checkpoint('expert/'))
    
    p_argmax = graph.get_tensor_by_name('actor_nn/max_act:0') 
    obs_ph = graph.get_tensor_by_name('obs:0') 

    def expert_policy(state):
        act = sess_expert.run(p_argmax, feed_dict={obs_ph:[state]})
        return np.squeeze(act)

    return expert_policy

def test_agent(policy, file_writer=None, test_games=10, step=0):
    game = FlappyBird()
    env = PLE(game, fps=30, display_screen=False)
    env.init()

    test_rewards = []
    for _ in range(test_games):
        env.reset_game()
        no_op(env)

        game_rew = 0

        while not env.game_over():

            state = flappy_game_state(env)

            action = 119 if policy(state) == 1 else None

            for _ in range(2):
                game_rew += env.act(action)

        test_rewards.append(game_rew)

        if file_writer is not None:
            summary = tf.Summary()
            summary.value.add(tag='test_performance', simple_value=game_rew)
            file_writer.add_summary(summary, step)
            file_writer.flush()

    return test_rewards


def DAgger(hidden_sizes=[32,32], dagger_iterations=20, p_lr=1e-3, step_iterations=1000, batch_size=128, train_epochs=20, obs_dim=8, act_dim=2):

    tf.reset_default_graph()

    ############################## EXPERT ###############################
    # load the expert and return a function that predict the expert action given a state
    expert_policy = expert()     
    print('Expert performance: ', np.mean(test_agent(expert_policy)))


    #################### LEARNER COMPUTATIONAL GRAPH ####################
    obs_ph = tf.placeholder(shape=(None, obs_dim), dtype=tf.float32, name='obs')
    act_ph = tf.placeholder(shape=(None,), dtype=tf.int32, name='act')

    # Multi-layer perceptron
    p_logits = mlp(obs_ph, hidden_sizes, act_dim, tf.nn.relu, last_activation=None)
        
    act_max = tf.math.argmax(p_logits, axis=1)
    act_onehot = tf.one_hot(act_ph, depth=act_dim)

    # softmax cross entropy loss
    p_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=act_onehot, logits=p_logits))
    # Adam optimizer
    p_opt = tf.train.AdamOptimizer(p_lr).minimize(p_loss)


    now = datetime.now()
    clock_time = "{}_{}.{}.{}".format(now.day, now.hour, now.minute, now.second)
    file_writer = tf.summary.FileWriter('log_dir/FlappyBird/DAgger_'+clock_time, tf.get_default_graph())

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    
    def learner_policy(state):
        action = sess.run(act_max, feed_dict={obs_ph:[state]})
        return np.squeeze(action)

    X = []
    y = []

    env = FlappyBird()

    env = PLE(env, fps=30, display_screen=False)
    env.init()    

    #################### DAgger iterations ####################
    
    for it in range(dagger_iterations):
        sess.run(tf.global_variables_initializer())
        env.reset_game()
        no_op(env)

        game_rew = 0
        rewards = []

        ###################### Populate the dataset #####################

        for _ in range(step_iterations):
            # get the current state from the environment
            state = flappy_game_state(env)

            # As the iterations continue use more and more actions sampled from the learner
            if np.random.rand() < (1 - it/5):
                action = expert_policy(state)
            else:
                action = learner_policy(state)

            action = 119 if action == 1 else None

            rew = env.act(action)
            rew += env.act(action)

            # Add the state and the expert action to the dataset
            X.append(state)
            y.append(expert_policy(state))

            game_rew += rew

            # Whenever the game stop, reset the environment and initailize the variables
            if env.game_over():
                env.reset_game()
                no_op(env)

                rewards.append(game_rew)
                game_rew = 0

        ##################### Training #####################

        # Calculate the number of minibatches
        n_batches = int(np.floor(len(X)/batch_size))

        # shuffle the dataset
        shuffle = np.arange(len(X))
        np.random.shuffle(shuffle)

        
        shuffled_X = np.array(X)[shuffle]
        shuffled_y = np.array(y)[shuffle]
        
        
        for _ in range(train_epochs):
            ep_loss = []
            # Train the model on each minibatch in the dataset
            for b in range(n_batches):
                p_start = b*batch_size

                # mini-batch training
                tr_loss, _ = sess.run([p_loss, p_opt], feed_dict={
                                obs_ph:shuffled_X[p_start:p_start+batch_size], 
                                act_ph:shuffled_y[p_start:p_start+batch_size]})

                ep_loss.append(tr_loss)
            
        agent_tests = test_agent(learner_policy, file_writer, step=len(X))

        print('Ep:', it, np.mean(ep_loss), 'Test:', np.mean(agent_tests))


    

if __name__ == "__main__":
    DAgger(hidden_sizes=[16,16], dagger_iterations=10, p_lr=1e-4, step_iterations=100, batch_size=50, train_epochs=2000)