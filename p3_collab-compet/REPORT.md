# Deep Reinforcement Learning
## Using Multiple Deep Deterministic Policy Agents
### Project's goal

In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.
The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, the agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

* After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
* This yields a single score for each episode.

The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.

### Learning Algorithm
Deep Deterministic Policy Agent, or DDPG, is an off-policy reinforcement learning algorithm targeted to environment with continous action space. DDPG is built with an Actor-Critic architecture where both the policy and the value function are estimated with a neural network.
DDPG being an off-policy uses both a local and a target network for the actor and the critic.

The environment is solved using a Multi-Agent DDPG or MADDPG, where standard DDPG agents are coordinated to learn collaboratively.
Each agent actor can only see its own environment observation, the critics however have access to the observations from all agents.

The Neural Network architecture chosen is as follow.

The actor network has two fully connected layers with 400 and 300 neurons respesctively. Each layer is passed through a ReLu non-linear activation function. 
The input of the first layer in the Agent's oberserved environment state.
The output of the first layer is normalized. The output layer is passed through a tanh activation function mapped to the 4 possible actions.

The critic network has two fully connected layers with 400 and 300 neurons respesctively. Each layer is passed through a ReLu non-linear activation function.

The input of the first layer is different from the actor as it also takes the action chosen by the actor.
The output of the first layer is normalized. The output layer is passed through a tanh activation function mapped to the 4 possible actions.

Both the Actor and the Critic are optimized using Adam Optimizer.

```python
# Defined on the MADDP Agent
BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 200        # minibatch size
GAMMA = 0.995           # discount factor of the future reward
TAU = 1e-3              # for soft update of target parameters θ_target = τ*θ_local + (1 - τ)*θ_target

UPDATE_EVERY = 4        # Update every UPDATE_EVERY episodes
UPDATE_COUNT = 3        # Number of update passes when updating

# Defined on the DDP Agent
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0      # L2 weight decay

MU = 0.                 # Ornstein-Uhlenbeck noise parameter
THETA = 0.15            # Ornstein-Uhlenbeck noise parameter
SIGMA = 0.2             # Ornstein-Uhlenbeck noise parameter
```

### Plot of rewards

### Ideas for future work
Beside hyper-parameter tuning (e.g. network size, noise, buffer size) other improvement should be explored:
* use a prioritized replay buffer by introducing a criterion to make importance experience being sampled more frequently rather than using a uniform sampling.
* use a different learning strategy across mulitple agents, e.g. average network weights accross agents or weighted updates based on the agents' scores

Other types of Agent algorithm could also be explored whislt keeping the Multi-Agent nature:
* MA-PPO
* Twin delayed DDPG
