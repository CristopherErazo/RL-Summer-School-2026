# Deep RL — Matteo Hessel (deep mind), 05/06/26

## Summary

The RL problem is the problem of an agent that has to learn in an environment in which is embedded. 
THe agent - env interact in a feedback loop where the agent performs actions and the environment returns observatons and rewards.

Agent behaviour = policy

We want to maximize the sum of discounted rewards, not just the inmediate rewards.

The env and agent are stochastic and therefore the reward is a random variable. We focus in expected rewards or value funcitons. 

In RL we want:
- values and policies learned in a model-free fashion directly from interacitons.
- use those interactions to learng a model of the environment.

RL solutions:

- learn policy directly - execute it
- learn values of (actions, states) - infer policy by inspecting
- learn a model of env - infer a policy by planning

Most RL algo will be at the intersection of at least 2 of the approaches. 

**All are functions**

- policy: $\pi: s \rightarrow  p(a)$
- values: $V : s \rightarrow V(s)$
- model: $m: s,a \rightarrow s',r$

We can use *function approximation*. DRL focuses in the case where the funciton is a large NN whose parameters are trained via GD on a suitable loss.

Obs: Deep RL is not the only answer, but it is very convenient. We can train NN very efficiently.

**First Simple Problem**

Prediction problem. Estimate state values of a given fixed policy $\pi$. Let the values be $v_\theta(s)$. *Deep Monte Carlo Prediction* trains $\theta$ to minimize:

$$L=(G_t - v_\theta(s_t))^2$$
$$\Delta \theta = -\nabla_\theta L = (G_t-v_\theta(s_t))\nabla_\theta v_\theta(s_t)$$

MC is unbiased but high variance! We can do temporal difference. 

*Deep TD(0) Prediction*
$$\Delta \theta =  (R_t+\gamma v_\theta(s_{t+1})-v_\theta(s_t))\nabla_\theta v_\theta(s_t)$$

Also Control: In this case we estimate values $q_\theta(s,a)$ of the optimal policy.

*Deep Q-learning update*
$$\Delta \theta =  (R_t+\gamma \max_{a'}q_\theta(s_{t+1},a')-q_\theta(s_t,a_t))\nabla_\theta q_\theta(s_t,a_t)$$


**Using Experience Replay**

Compute Q-learning loss not in the most recent trancisions but in old ones sampled from a large memory buffer. 
- larger buffer -> less correlated samples
- reuse transitions
- can prioritize novel
- compute updates from batches

**Generalization in Deep RL**

Gen. is updating belief about some state in responce of observing information about different states.

Gen. is the focus of supervised learning. We want to maximize performance in a test set. 

In supervised learning generalization is improved:
- explicitly: via validation data and early stopping
- implicitly: via regularization, dropout


How to consider genrealization in deep RL?

-There can e generalization error where the target we want to predict is not smooth since NN are smooth and there tent to be a leakage effect. 

Deadly triad for generalization errors: Function approximation + bootstraping + off-policy.

A common remedy is to use target networks which are a copy of the network that u use to compute the values and just update every some steps to break the feedback loop

**State Representations**

Use the state representations to different tasks:

- Define pseudo rewards based on representations.
- Auxiliar tasks to enrich representations. 
- inmediate reward prediction
- next state prediction
- Learn at multiple time scales


**Distribution DRL**

We can try to predict the entire reward distribution, not just the expected reward. 




## Papers & References
-

## Questions I Still Have
-

