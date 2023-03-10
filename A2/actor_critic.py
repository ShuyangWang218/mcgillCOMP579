import numpy as np
import random
from operator import add

class ActorCritic:
    def __init__(self, env, alpha, gamma, epsilon, num_episodes,num_bins,):
        self.env = env
        self.alpha = alpha
        self.epsilon = epsilon
        self.gamma = gamma
        self.num_bins = num_bins
        self.num_actions = env.action_space.n
        self.lower_bounds=env.observation_space.low
        self.upper_bounds=env.observation_space.high
        self.num_episodes = num_episodes

        self.num_states = self.num_bins**env.observation_space.shape[0]
        self.actor_weights = np.random.uniform(low=-0.001, high=0.001,
                                               size=(num_bins, num_bins, num_bins, num_bins, self.num_actions))
        self.critic_weights = np.random.uniform(low=-0.001, high=0.001,
                                                size=(num_bins, num_bins, num_bins, num_bins))

        # this list stores sum of rewards in every learning episode
        self.sum_episode_rewards=[]

    '''
    def create_bins(self):
        state_bins = []
        for i in range(self.env.observation_space.shape[0]):
            low, high = self.env.observation_space.low[i], self.env.observation_space.high[i]
            state_bins.append(np.linspace(low, high, self.num_bins + 1)[1:-1])
        return state_bins
    '''

    def discretize_state(self, state):
        '''
        state_idx = 0
        for i in range(self.env.observation_space.shape[0]):
            state_idx += np.digitize(state[i], self.state_bins[i]) * (self.num_bins ** i)
        return state_idx
        '''
        '''This function takes state, the list of exact values of position, velocity, angle and
        angular velocity as input, and returns the indices of the corresponding bins, respectively.
        '''
        state_idx = []
        for i in range(self.env.observation_space.shape[0]):
            bins = np.linspace(self.lower_bounds[i], self.upper_bounds[i], self.num_bins)
            idx = np.maximum(np.digitize(state[i], bins) - 1, 0)
            state_idx.append(idx)
        return tuple(state_idx)

    def discretize_state2(self, state):
        '''This function takes state, the list of exact values of position, velocity, angle and
        angular velocity as input, and returns the indices of the corresponding bins, respectively.
        '''
        position = state[0]
        velocity = state[1]
        angle = state[2]
        angularVelocity = state[3]

        cartPositionBin = np.linspace(self.lowerBounds[0], self.upperBounds[0], self.numberOfBins)
        cartVelocityBin = np.linspace(self.lowerBounds[1], self.upperBounds[1], self.numberOfBins)
        poleAngleBin = np.linspace(self.lowerBounds[2], self.upperBounds[2], self.numberOfBins)
        poleAngleVelocityBin = np.linspace(self.lowerBounds[3], self.upperBounds[3], self.numberOfBins)

        indexPosition = np.maximum(np.digitize(position, cartPositionBin) - 1, 0)
        indexVelocity = np.maximum(np.digitize(velocity, cartVelocityBin) - 1, 0)
        indexAngle = np.maximum(np.digitize(angle, poleAngleBin) - 1, 0)
        indexAngularVelocity = np.maximum(np.digitize(angularVelocity, poleAngleVelocityBin) - 1, 0)

        return tuple([indexPosition, indexVelocity, indexAngle, indexAngularVelocity])

    def choose_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()
        else:
            #state_idx = self.discretize_state(state)
            #a = self.actor_weights[state_idx]
            action_probs = self.softmax(self.actor_weights[state_idx])
            return np.random.choice(self.num_actions, p=action_probs)

    def update(self, state_idx, action, reward, next_state_idx):
        #state_idx = self.discretize_state(state)
        next_state_idx = self.discretize_state(next_state_idx)
        td_error = reward + self.gamma * self.critic_weights[next_state_idx] - self.critic_weights[state_idx]
        self.critic_weights[state_idx] += self.alpha * td_error
        action_probs = self.softmax(self.actor_weights[state_idx])
        one_hot_action = np.zeros(self.num_actions)
        one_hot_action[action] = 1
        #temp1 = np.outer((one_hot_action - action_probs), state_idx)
        temp = self.alpha * td_error * np.outer((one_hot_action - action_probs), state_idx)
        a = self.actor_weights[state_idx+(action,)]
        self.actor_weights[state_idx+(action,)] += self.alpha * td_error

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)

    def simulateEpisodes(self):
        for i in range(self.num_episodes):
            #state = self.env.reset()
            (state, _) = self.env.reset()
            state = list(state)
            episode_reward = 0

            terminal_state = False
            while not terminal_state:
                #get current action proposal
                state_idx = self.discretize_state(state)
                action = self.choose_action(state_idx)
                (state_prime, reward, terminal_state, _, _) = self.env.step(action)
                episode_reward += reward

                state_prime = list(state_prime)
                state_prime_idx = self.discretize_state(state_prime)

                if not terminal_state:
                    td_error = reward + self.gamma * self.critic_weights[state_prime_idx] - self.critic_weights[state_idx]
                    if td_error > 0:
                        self.update(state_idx, action, reward, state_prime_idx)

                state = state_prime

            self.sum_episode_rewards.append(episode_reward)


