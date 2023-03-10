import gymnasium as gym
import numpy as np
import time

# import the class that implements the Q-Learning algorithm
from q_learning import Q_Learning
from actor_critic import ActorCritic
from plot_learning_curves import plot_learning_curve as plc

#env=gym.make('CartPole-v1',render_mode='human')
env=gym.make('CartPole-v1', render_mode='rgb_array')

# here define the parameters for state discretization
numberOfBins=10

# define the parameters
runs=10
alpha_list = [1/4, 1/8, 1/16]
gamma = 1
epsilon_list = [0.1, 0.2, 0.3]
numberEpisodes = 1000

reward_list = np.empty([runs, numberEpisodes])
n = len(alpha_list)
average_list = np.zeros([n,numberEpisodes])
std_error_list = np.zeros([n,numberEpisodes])


#Q-learning
print("Executing Q-learning algorithm...")
(state,_)=env.reset()
for epsilon in epsilon_list:
    print("\nepsilon = " + str(epsilon))
    for ida, alpha in enumerate(alpha_list):
        start = time.time()
        for run in range(runs):
            np.random.seed(run)
            Q=Q_Learning(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins)
            # run the Q-Learning algorithm
            Q.simulateEpisodes()
            reward_list[run] = Q.sumRewardsEpisode

        average_list[ida] = np.mean(reward_list, axis=0)
        std_error_list[ida] = np.std(reward_list, axis=0)/np.sqrt(runs)
        end = time.time()
        print("time (alpha = %s): %f" %(str(alpha), end - start))
    plc(average_list[0],average_list[1],average_list[2], std_error_list[0], std_error_list[1],
                           std_error_list[2],epsilon, numberEpisodes,"Q-learning")


'''

#Actor-cirtic
print("Executing actor-critic algorithm...")
(state,_)=env.reset()
for epsilon in epsilon_list:
    print("\nepsilon = " + str(epsilon))
    for ida, alpha in enumerate(alpha_list):
        start = time.time()
        for run in range(runs):
            np.random.seed(run)
            Q=ActorCritic(env,alpha,gamma,epsilon,numberEpisodes,numberOfBins)
            # run the Q-Learning algorithm
            Q.simulateEpisodes()
            reward_list[run] = Q.sum_episode_rewards

        average_list[ida] = np.mean(reward_list, axis=0)
        std_error_list[ida] = np.std(reward_list, axis=0)/np.sqrt(runs)
        end = time.time()
        print("time (alpha = %s): %f" %(str(alpha), end - start))
    plc(average_list[0],average_list[1],average_list[2], std_error_list[0], std_error_list[1],
                           std_error_list[2],epsilon, numberEpisodes,"Actor-critic")


'''

# close the environment
env.close()

print('done')