
# Project 1: Navigation
Udacity Deep Reinforcement Learning Nanodegree

## Deep Q-Network Agent learning on a Unity environment

### Goal 
Build and train a reinforcement learning agent using a Deep Q-Network architecture to collect bananas in a [Unity environment](https://github.com/Unity-Technologies/ml-agents/).

The reward scheme is defined as follow, the agent is rewarded by 1 point for each yellow banana collected and by -1 for each blue banana.
The performance of the agent is assessed over 100 consecutive episodes, the agent is considered to have solved the environment if he achieves a score of 13 or more.

### See it in Action
Check the notebook [Navigation-solution.ipynb](p1_navigation/Navigation-solution.ipynb) to see how the agent is instantiated and trained.
<img src="p1_navigation/Unity-Bananas.gif" width="75%" alt="Unity Banana Agent" title="Unity Banana Agent" />

*Captured with OBS Studio*

### Environment & Design
This particular instance was developed on Win10 with unity 0.4

### Design & Approach
The agent is a Deep Q-Network agent with experience replay.

### Future improvements
TODO
