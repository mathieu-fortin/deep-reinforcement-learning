# Deep Reinforcement Learning
# Using a Deep Deterministic Policy Agent

## Project's goal
The goal in this environment is to control a double-jointed arm and move it to target locations highlited by a green sphere.
A reward of +0.1 is provided for each step that the agent's hand is in the goal location.
Thus, the goal of the agent is to maintain its position at the target location for as many time steps as possible.

## Learning Algorithm
Deep Deterministic Policy Agent, or DDPG, is an off-policy reinforcement learning algorithm targeted to environment with continous action space.
DDPG is built with an Actor-Critic architecture where both the policy and the value function are estimated with a neural network.
DDPG being an off-policy uses both a local and a target network for the actor and the critic.

```python
BUFFER_SIZE = int(1e6)  # replay buffer size
BATCH_SIZE = 128        # minibatch size
GAMMA = 0.99            # discount factor of the future reward
TAU = 1e-3              # for soft update of target parameters θ_target = τ*θ_local + (1 - τ)*θ_target
LR_ACTOR = 1e-3         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0.0        # L2 weight decay

UPDATE_EVERY = 20       # Update every UPDATE_EVERY steps
UPDATE_COUNT = 10       # Number of update passes when updating
```

The Neural Network architecture chosen is as follow.

The actor network has two fully connected layers with 256 and 128 neurons respesctively. Each layer is passed through a ReLu non-linear activation function.
The output layer is passed through a tanh activation function mapped to the 4 possible actions.

The critic network has two fully connected layers with 256 and 128 neurons respesctively. Each layer is passed through a leaky ReLu non-linear activation function.
The choice of leaky ReLu here is essentially to mitigate the impact of vanishing gradient and the dying ReLu effect.

Both the Actor and the Critic are optimized using Adam Optimizer.


The environment is solved using 20 simultaneous agents, all agents share the same ReplayBuffer from which experiences are sampled.
At the end of each episode the network weights of the best agent is broadcasted to the other agents.

## Plot of rewards

## Ideas for future work
Besides using other learning algorithm such as PPO, D4PG, A3C; several attemps can be made to further improve the performence of DDPG:
* We could make use of a prioritized replay buffer by introducing a criterion to make importance experience being sampled more frequently rather than using a uniform sampling.
* Different architectures with multiple agents could be compared, adversarial vs collaborative agents by switching the selection of the best agents with a averaging or weighting of the agents' network weights.

