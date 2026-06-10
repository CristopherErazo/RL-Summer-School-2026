# Multi-agent RL — Giorgia Ramponi, 10/06/26

## Summary

A MDP is characterized by a: state space, action space, envirnoment model, discount factor (horizon), and reward function. Here we want to maximize the expected discounted reward.

In multi agent RL, the set of actions encompases a set of actions for each agent and the reward function of each agent depends on the state and the joint set of actions of all the agents and therefore there is not a clear objective measure that we want to maximize.


**Normal Form Game**

Is the simplest multi agent setting where there are not states and no transitions. The game is fully specified by a reward funciton that depends on the actions of the agents. 

$$R_i \in \mathbb{R}^{|A_1|\times|A_2|}\times \cdots$$

**Zero sum games** (case of two players game): a pure adversarial setting where $R_1 = -R_2$.

**Nash Equilibrium**

In multi agents the notion of optimallity is the Nash equilibrium (multi-agent counterpart of Bellman optimallity equation). 
Is a fixed point where no agent can gain by unilateraly deviating from it.


...

**Simple or complicated settings**

Even in zero sum 2 player games the problem is already complicated even if the agents have full knowledge of the game and the objective of the other agents. In general you can have information assymetry, partial observability, non-stationarity, etc.


## Papers & References
-

## Questions I Still Have
-

