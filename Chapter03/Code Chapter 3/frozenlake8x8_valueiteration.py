import numpy as np
import gym

def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def value_iteration(eps=0.0001):
    '''
    Value iteration algorithm
    '''
    V = np.zeros(nS)
    it = 0

    while True:
        delta = 0
        # update the value of each state using as "policy" the max operator
        for s in range(nS):
            old_v = V[s]
            V[s] = np.max([eval_state_action(V, s, a) for a in range(nA)])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break
        else:
            print('Iter:', it, ' delta:', np.round(delta, 5))
        it += 1

    return V

def run_episodes(env, V, num_games=100):
    '''
    Run some test games
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            action = np.argmax([eval_state_action(V, state, a) for a in range(nA)])
            next_state, reward, done, _ = env.step(action)

            state = next_state
            tot_rew += reward 
            if done:
                state = env.reset()

    print('Won %i of %i games!'%(tot_rew, num_games))

            
if __name__ == '__main__':
    # create the environment
    env = gym.make('FrozenLake-v0')
    # enwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n

    # Value iteration
    V = value_iteration(eps=0.0001)
    # test the value function on 100 games
    run_episodes(env, V, 100)
    # print the state values
    print(V.reshape((4,4)))

