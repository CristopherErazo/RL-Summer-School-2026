# Stochastic Bandits — Tor Lattimore, 08/06/26

## Summary

We want to have a good strategy to interact with 'bandits'. Historically the applications are *medical trials*.

U want to allocate the drugs in such a way that u save as many patients as possible. 

Bandits are the simplest type of RL. Is a single state MDP, simple model to test ideas. 

**Formal Model**

- $k$ actions
- For each action there is an unknonw mean $\mu \in [0,1]^k$
- Known time horizon
- Learner and environments interact over n rounds
- Actions are $A_1,\cdots,A_n$
- Rewards are $R_1,\cdots,R_n$

**Exploration - Explotation Dilemma**

How to decide between a known quantity and an  unknown one?

- Should I try a new coffee shop?
- Should I bring my climbing shoes to Milano?
- Should I explore a new line of research?

Bandits capture the escence of this problems. 

**Objectives**

Expected cumulative regret: every time you play an action you wish u could've done the optimal action and there is a regret for it:

$$Reg_n = \mathbb{E} \sum_{t=1}^n (\mu_1 - \mu_{A_t})=  \mathbb{E} \sum_{t=1}^n \Delta_{A_t}$$

Assume (1) is the best bandit.

- *Optimism Principle:* Acts as if you are the nicest plausibly possible environment. 
- We got empirical means of each bandit $\nu$


The idea for an algorithm would be to estimate for each bandit the mean and the upper level of the confidence bound and we pick the action with the largest mean+bonus. 

This is due to the *Hoeffding's bound*:

Suppose that $X_1,\cdots,X_n$ are $[0,1]$ valued iid rv with mean $\mu$ , then for any $\delta \in (0,1)$,

$$\mathbb{P}\left( |\hat\mu - \mu| \geq \sqrt{\frac{\log(2/\delta)}{2n}} \right)\leq\delta$$

Challenge: come up with tighters confidence bounds.

*RL is different from classical statistics because the data I gattered before conditiones the next actions and therefore the data is not exactly IID.*

We can rethink a bit the bandit problem. Assume you draw all the sequence of rewards of the bandits iid, then the way in which I *uncover* the rewards of each bandit is no-iid, byt now i have a sequence of iid variables over which I can work. 

- $\hat\mu_a(t):$ mean of the bandit $a$ after $t$ times. (non iid)
- $\hat\mu_{at}:$ mean I would get if I played bandit $a$, $t$ times. (iid)


**Policy Gradient for Stochastic Bandits**



## Papers & References
-

## Questions I Still Have
-

