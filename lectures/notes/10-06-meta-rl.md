# Meta RL — Aviv Tamar, 10/06/26

## Summary

We want agents that solve complex task, the problem is that tasks are very messy and we cannot predict in advance all the scenarios. We need agents that can adapt and generalize to scenarios they haven't seen before (main problem in robotics nowadays). 

*RL looks like the right approach for 'robots' that can solve any task.*

The problem is that we need the agent to adapt way faster than what RL allows. We need to give more prior knowledge. 

**How to use prior data to make agents learn much faster:** This has different names in the literature eg. meta learning. 

**Question**

What can we learn from training tasks?

* In supervised learning (eg classification): we learn a decision rule.
* In reinforcement learning -> ? generalization in RL can be very different than in SL. Use data to learn to explore?

**Exploration in RL**

* If no prior information -> 'dont waste time in non-interesting states' (regret bounds, PAC bounds,...)
* If we have a prior over MDPs -> plan to obtain information/rewards (well defined optimal exploration)

**Bayesian RL (META RL)**

* multi task RL
* Each task MDP $M = \{P(s'|s,a),r(s)\}$
* Task distribution $P(M)$

$$Goal: \; \max_\pi\mathbb{E}_{M\sim P(M)} \mathbb{E}_{M} \sum_{t=0}^{H} r_t $$

It ends up being a special case of a Partially Observable MDP because the agent does not know in which MDP it was thrown into. 

**Partially Observable MDP**

* states $s$, actions $a$.
* transitions $P(s'|s,a)$
* observations $P(o|s)$
* history $h_t = (o_0,a_0,r_0,\cdots,o_t)$
* policy $\pi(a|h)$ *not markovian?*
* reward $r(s)$

Good news: POMDP is MDP over histories.

$h_{t+1} = \{h_t,a_t,r_t,o_{t+1}\} \rightarrow P(h_{t+1} | h_t,a_t,r_t)$ is well defined.

Bad news: History space grows exponentially with $H$.


**Information states**

Are summary statistics of the history. 
$$z_t = f_t(h_t)$$

We can use dynamic programming in the information states. 

Examples:

* Obvious : $z_t=h_t$
* Belief : $z_t =  P(s_t | h_t)$

Main insight: World Models! -> LSTM, Transformers, VAE, Classifiers. (self supervised learning with the trajectory/history)

**Back to MRL**

OBS: $P(M) \rightarrow$ set of training / test tasks/


**Parametric Formulation**

We assume a parametric space $\Theta$ and an MDP mapping $g(\theta) : \Theta \rightarrow M$ with a parameter distribution $P(\theta)$.

The loss is $L(\pi)$ and the Bayes optimal policy $\pi_{BP}(h_t)$.

We are given $N$ simulators: $\theta_1,\cdots,\theta_N$.

Find $\pi$ with lowest regret: $R(\pi) = L(\pi_{BO}) - L(\pi)$.

Essentially: approximate $P(\theta)$ from samples.










## Papers & References
-

## Questions I Still Have
-

