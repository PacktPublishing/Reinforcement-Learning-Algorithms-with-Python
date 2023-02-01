import numpy as np
import gym

def eval_state_action(V, s, a, gamma=0.99):
    return np.sum([p * (rew + gamma*V[next_s]) for p, next_s, rew, _ in env.P[s][a]])

def policy_evaluation(V, policy, eps=0.0001):
    '''
    Policy evaluation. Update the value function until it reach a steady state
    '''
    while True:
        delta = 0
        # loop over all states
        for s in range(nS):
            old_v = V[s]
            # update V[s] using the Bellman equation
            V[s] = eval_state_action(V, s, policy[s])
            delta = max(delta, np.abs(old_v - V[s]))

        if delta < eps:
            break

def policy_improvement(V, policy):
    '''
    Policy improvement. Update the policy based on the value function
    '''
    policy_stable = True
    for s in range(nS):
        old_a = policy[s]
        # update the policy with the action that bring to the highest state value
        policy[s] = np.argmax([eval_state_action(V, s, a) for a in range(nA)])
        if old_a != policy[s]: 
            policy_stable = False

    return policy_stable


def run_episodes(env, policy, num_games=100):
    '''
    Run some games to test a policy
    '''
    tot_rew = 0
    state = env.reset()

    for _ in range(num_games):
        done = False
        while not done:
            # select the action accordingly to the policy
#             print(env.step(policy[state]))
#             break
            next_state, reward, done, _, _ = env.step(policy[state])
                
            state = next_state
            tot_rew += reward 
            if done:
                state = env.reset()

    print('Won %i of %i games!'%(tot_rew, num_games))
            
if __name__ == '__main__':
    # create the environment
    env = gym.make('FrozenLake-v1')
    # enwrap it to have additional information from it
    env = env.unwrapped

    # spaces dimension
    nA = env.action_space.n
    nS = env.observation_space.n
    
    # initializing value function and policy
    V = np.zeros(nS)
    policy = np.zeros(nS)

    # some useful variable
    policy_stable = False
    it = 0

    while not policy_stable:
        policy_evaluation(V, policy)
        policy_stable = policy_improvement(V, policy)
        it += 1

    print('Converged after %i policy iterations'%(it))
    run_episodes(env, policy)
    print(V.reshape((4,4)))
    print(policy.reshape((4,4)))
