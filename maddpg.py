# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from ddpg import DDPGAgent, ReplayBuffer, OUNoise
from model import Critic
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = 'cpu'

#BATCH_SIZE = 200
BATCH_SIZE = 600
BUFFER_SIZE = int(1e7)
RANDOM_SEED = 1
GAMMA = 0.95

#Code for MADDPG algorithm
class MADDPG:
    def __init__(self, discount_factor=0.95, tau=0.02):
        super(MADDPG, self).__init__()


        #self.critic_local = Critic(state_size = 24 * 2,  action_size = 2 * 2, seed = 1).to(device)
        #self.critic_target = Critic(state_size = 24 * 2,  action_size = 2 * 2, seed = 1).to(device)

        #Agent takes in id, state size, action size and random seed
        #For tennis, obersvation space is 8, stacked obserrvation 3, action space 2
        self.maddpg_agent = [DDPGAgent(id = 0, state_size = 24,  action_size = 2, random_seed = 1),
                             DDPGAgent(id = 1, state_size = 24,  action_size = 2, random_seed = 1)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.iter = 0

        # Replay memory
        self.memory = ReplayBuffer(action_size = 2, buffer_size = BUFFER_SIZE, batch_size = BATCH_SIZE, seed = RANDOM_SEED)
        self.num_agents = 2
        self.action_size =2
        self.observation_size = 24

    def act(self, obs_all_agents, noise):
        """get actions from all agents in the MADDPG object"""

        actions = []
        for agent, state in zip(self.maddpg_agent, obs_all_agents):
            #actions[i] = agent.act(obs_all_agents[i], add_noise=noise)
            action = agent.act(state, add_noise=noise)
            actions.append(action)
        actions = np.array(actions).reshape(1, -1)
        #print(actions)
        return actions


    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        all_states = states.reshape(1, -1) # change 2x24 to 1x48
        all_next_states = next_states.reshape(1, -1) #change 2x24 to 1x48

        # Save experience / reward
        self.memory.add(all_states, actions, rewards, all_next_states, dones)

        # Learn, if enough samples are available in memory
        if len(self.memory) > BATCH_SIZE:
            experiences = [self.memory.sample() for _ in range(2)]
            self.learn(experiences)

    def learn(self, experiences):
        """agent learn from random sample from buffer."""
        all_actions = []
        all_next_actions = []
        #print("MADDPG Learn")
        #get next_action, action for each agent
        for i, agent in enumerate(self.maddpg_agent):
             #experiences = self.memory.sample()
             states, _, _, next_states, _ = experiences[i]
             index = torch.tensor([i]).to(device)
             state = states.reshape(-1, 2, 24).index_select(1, index).squeeze(1)
             tmp_action = agent.actor_local(state)
             all_actions.append(tmp_action)

             next_state = next_states.reshape(-1, 2, 24).index_select(1, index).squeeze(1)
             tmp_next_action = agent.actor_target(next_state)
             all_next_actions.append(tmp_next_action)

        #train agent
        for i, agent in enumerate(self.maddpg_agent):
            agent.learn(experiences[i], GAMMA, all_actions, all_next_actions, i)

    def save(self):
        """save networks."""
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor_local.state_dict(), 'checkpoint_actor_{}.pth'.format(i))
            torch.save(agent.critic_local.state_dict(), 'checkpoint_critic_{}.pth'.format(i))

    def load(self):
        """load networks."""
        for i, agent in enumerate(self.maddpg_agent):
            agent.actor_local.load_state_dict(torch.load('checkpoint_actor_{}.pth'.format(i)))
            agent.critic_local.load_state_dict(torch.load('checkpoint_critic_{}.pth'.format(i)))

    def reset(self):
        """reset networks."""
        for agent in self.maddpg_agent:
            agent.reset()
