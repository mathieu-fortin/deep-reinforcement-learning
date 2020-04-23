import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import namedtuple, deque

from ddpg_agent import Agent
from memory import ReplayBuffer

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 200        # minibatch size
GAMMA = 0.995           # discount factor
TAU = 1e-3              # for soft update of target parameters

UPDATE_EVERY = 4        # Update every UPDATE_EVERY episodes

NOISE = 1.0             # Initial Noise Amplitude
NOISE_DECAY = 1         # Noise decay    

CLIP_GRADIENT = False

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Maddpg():
    """Interacts with and learns from the environment with multiple ddpg agents."""
    
    def __init__(self, state_size, action_size, num_agents, random_seed, tau=TAU):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            num_agents (int): number of ddpg agents
            random_seed (int): random seed
        """
        super(Maddpg, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)
        self.tau = tau

        # List of controlled DDPG Agents
        self.num_agents = num_agents
        self.agents = [Agent(state_size, action_size, random_seed, num_agents, device) for _ in range(num_agents)]
        
        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed)
        
    def step(self, states, actions, rewards, next_states, dones, time_step):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        # Save experience / reward
        #self.memory.add(state, states, actions, rewards, next_state, next_states, dones)
        self.memory.add(states.flatten(), actions.flatten(), rewards, next_states.flatten(), dones)
        
        # Learn, if enough samples are available in memory
        # Learn every UPDATE_EVERY time steps.
        if len(self.memory) > BATCH_SIZE and (time_step % UPDATE_EVERY == 0):
            for _ in range(UPDATE_EVERY-1): #range(self.num_agents):
                for agent_number, agent in enumerate(self.agents):
                    experiences = self.memory.sample(device)
                    self.learn(agent_number, experiences, GAMMA)

    def act(self, states, noise):
        """Returns agents' actions for the given states as per current policy."""
        return np.array([agent.act(state, noise) for agent, state in zip(self.agents, states)]).reshape(1,-1).squeeze()

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def learn(self, agent_number, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        #states, actions, rewards, next_states, dones = map(transpose_to_tensor, experiences)
        all_states, all_actions, rewards, all_next_states, dones = experiences
        agent = self.agents[agent_number]
        
        all_states = all_states.T.reshape(self.num_agents, self.state_size, -1) # rows contain transposed states
        all_actions = all_actions.T.reshape(self.num_agents, self.action_size,-1)
        all_next_states = all_next_states.T.reshape(self.num_agents, self.state_size, -1)
        
        states = all_states[agent_number].T
        actions = all_actions[agent_number].T
        next_states = all_next_states[agent_number].T
        
        other_agents = torch.Tensor(list(set(range(self.num_agents)) - set([agent_number]))).long().to(device)
        other_states = all_states.index_select(0, other_agents).reshape(-1, self.state_size)
        other_actions = all_actions.index_select(0, other_agents).reshape(-1, self.action_size)
        other_next_states = all_next_states.index_select(0, other_agents).reshape(-1, self.state_size)
        
        # Reorder the agents information with current agent first
        reordered_agents = torch.cat((torch.Tensor([agent_number]).long().to(device), other_agents))
        all_states = all_states.index_select(0, reordered_agents).reshape(self.num_agents, self.state_size, -1)
        all_actions = all_actions.index_select(0, reordered_agents).reshape(self.num_agents, self.action_size, -1)
        all_actions = all_actions.reshape(self.action_size*self.num_agents, -1).T
        all_next_states = all_next_states.index_select(0, reordered_agents).reshape(self.num_agents, self.state_size, -1)
        all_next_states = all_next_states.reshape(self.state_size * self.num_agents, -1).T
        
        # ---------------------------- update critic ---------------------------- #        
        all_actions_next = torch.cat(list(map(lambda x: agent.actor_target(x.T), all_states)), dim=1).to(device)
        
        # Get predicted next-state actions and Q values from target models
        Q_targets_next = agent.critic_target(all_next_states, all_actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = agent.critic_local(all_states.reshape(self.state_size* self.num_agents, -1).T, all_actions)
        
        # broadcast to match target shape and remove warning message
        Q_expected = torch.cat((Q_expected, Q_expected), dim=1).to(device)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        if CLIP_GRADIENT:
            torch.nn.utils.clip_grad_norm_(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        action_states = list(map(lambda x: agent.actor_local(x.T), all_states))
        actions_pred = torch.cat(action_states, dim = 1).to(device)
        actor_loss = -agent.critic_local(all_states.reshape(self.state_size* self.num_agents, -1).T, actions_pred).mean()
        
        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target, TAU)
        agent.soft_update(agent.actor_local, agent.actor_target, TAU)                     

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for agent in self.agents:
            soft_update(agent.target_actor, agent.actor, self.tau)
            soft_update(agent.target_critic, agent.critic, self.tau)
    
    def save(self):
        for i, agent in enumerate(self.agents):
            actor_local_filename = "model/actor_local_{}.pth".format(i)
            critic_local_filename = "model/critic_local_{}.pth".format(i)
            actor_target_filename = "model/actor_target_{}.pth".format(i)
            critic_target_filename = "model/critic_target_{}.pth".format(i)
            torch.save(agent.actor_local.state_dict(), actor_local_filename) 
            torch.save(agent.critic_local.state_dict(), critic_local_filename)             
            torch.save(agent.actor_target.state_dict(), actor_target_filename) 
            torch.save(agent.critic_target.state_dict(), critic_target_filename)
            