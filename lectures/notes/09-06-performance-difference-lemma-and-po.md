# Performance Difference Lemma and PO Methods — Gergely Neu, 09/06/26

## Summary


**Markov Decisions Processes**

Modelling tool for modelling the interactions of an agent with an environment. 

- Markov property: $X_{t+1}$ only depends on $(X_t,A_t)$
- Stationarity: $P(\cdot|X_t,A_t)$ does not depend on $t$.

Therefore enough to consider stationary policies: $\pi(a|x) = \mathbb{P}(A_t=a|X_t=x)$

Solving MDPs in the canonical way implies solving the Bellman optimiality equations.

**Goal Today:** study alternative theory that can serve as foundation to policy - based RL methods.

**Occupancy Measures**

Motivation: the distounted reward of a policy is 
$$\rho^\pi = \mathbb{E}_\pi\left[\sum_{t=0}^\infty \gamma^t r(X_t,A_t)\right] $$


## Papers & References
-

## Questions I Still Have
-

