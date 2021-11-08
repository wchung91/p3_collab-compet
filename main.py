from unityagents import UnityEnvironment
import numpy as np
from maddpg import MADDPG
env = UnityEnvironment(file_name="./Tennis_Linux_NoVis/Tennis.x86_64")
from collections import deque
import matplotlib.pyplot as plt
import random

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]


# reset the environment
env_info = env.reset(train_mode=True)[brain_name]

# number of agents
num_agents = len(env_info.agents)
print('Number of agents:', num_agents)

# size of each action
action_size = brain.vector_action_space_size
print('Size of each action:', action_size)

# examine the state space
states = env_info.vector_observations
state_size = states.shape[1]
print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
print('The state for the first agent looks like:', states[0])


# examine the state space
n_episodes = 100000     #Number of episodes
max_t = 1000000         #Number of steps
print_every=100         #when to print
Train = 0               #1 to train, 0 to test

#Initialize MADDPG agent
maddpg = MADDPG()

#Start of training
if Train == 1:
    scores_deque = deque(maxlen=print_every)                    # queue for averaging 100 episodes
    scores_history = []
    for i_episode in range(1, n_episodes+1):                    # play game for n_episodes episodes
        env_info = env.reset(train_mode=True)[brain_name]       # reset the environment
        states = env_info.vector_observations                   # get the current state (for each agent)
        scores = []                                             # initialize the score (for each agent)
        for t in range(max_t):                                  # Start step

            actions = maddpg.act(states, noise=True)            #get actions for each agent
            env_info = env.step(actions)[brain_name]            # send all actions to tne environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            maddpg.step(states, actions, rewards, next_states, dones) #update the agents
            scores.append(rewards)
            states = next_states                                # roll over states to next time step
            if np.any(dones):                                   # exit loop if episode finished
                break

        #get the max between agents
        max_score= np.max(np.sum(np.array(scores),axis=0))
        #print(max_score)
        scores_deque.append(max_score)                          # Add max sore to queue
        scores_history.append(max_score)                        # Add max score to history list
        #print('\rEpisode {} Score: {:.2f}'.format(i_episode, max_score))

        #print average score
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

            #save agent if score is above 5.5
            if np.mean(scores_deque) > 1.0:
                maddpg.save()
                break
    #code for saving figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_history)+1), scores_history)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig("ScoresTraining")

#Start of testing agents
if Train == 0:
    scores_deque = deque(maxlen=print_every)                    # queue for averaging 100 episodes
    scores_history = []                                         # list of score history

    maddpg.load()                                               # load the agent's networks
    n_episodes = 100
    for i_episode in range(1, n_episodes+1):                    # play game for 100 episodes
        env_info = env.reset(train_mode=False)[brain_name]      # reset the environment
        states = env_info.vector_observations                   # get the current state (for each agent)
        scores = []                                             # Initialize list
        for t in range(max_t):                                  # start step
            actions = maddpg.act(states, noise=False)           # get actions from agents
            env_info = env.step(actions)[brain_name]            # send all actions to tne environment
            next_states = env_info.vector_observations          # get next state (for each agent)
            rewards = env_info.rewards                          # get reward (for each agent)
            dones = env_info.local_done                         # see if episode finished
            #maddpg.step(states, actions, rewards, next_states, dones)
            scores.append(rewards)                              # store reward
            states = next_states                                # roll over states to next time step
            if np.any(dones):                                   # exit loop if episode finished
                break
                #Print average score every 100 step
        max_score= np.max(np.sum(np.array(scores),axis=0))      # get the max score between agents
        #print(max_score)
        scores_deque.append(max_score)                          # store max score to queue
        scores_history.append(max_score)                        # store max score to history
        print('\rEpisode {} Score: {:.2f}'.format(i_episode, max_score))    #print score for every episode

        #print average
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))

    #code for saving figure
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores_history)+1), scores_history)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    #plt.show()
    plt.savefig("TrainedScore")



    #print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))
env.close()
