# MDPs & Dynamic Programming — Leif Döring, 03/06/26 (9:00 am - 10:30 am)

## Summary

MDPs are not a very well defined object across the literature.

Idea of a decision model:
- An agent has to make a decision. 2 step experiment
- 1 step: choose an action (or sample it)
- 2 step: get a reward 
- Is possible to optimize the reward under the choice of actions. 

Sequencial decisions appear a lot in many problems and they are generally hard because we don't optimize a single step but the full trajectory.


The mathematical framework to work on is Markov Chains: stochastic process that satifies the Markov Property. 

Markov Reward Chains: Stochastic process $S_0,R_0,S_1,R_1,\cdots$ drawn from $p(s',r|s)$. It is not a 2-dim Markov process, is just Markov on states.

Final Elements (environment):
- State Space $\mathcal{S}$
- Set of actions $\mathcal{A_s}$ (assume $\mathcal{A}$) 
- Kernel: $p : \mathcal{S}\times\mathcal{A} \rightarrow \mathcal{S} \times \mathcal{R} \qquad p(s',a|s,r)$ 


Bringing agents into the game: the agent will be defined by a policy $\pi(a|s)$. At every step the agent and environment interact. 

**DEFINITION: Environment $p$ + Policy $\pi$ + Initial Distribution $\nu$ = Markov Decision Process.**
We define a kernel as: 
$$q((s',a'),r | (s,a)) = p(s', r|s,a)\pi(a'|s')$$

**Optimization Problem - Policy Evaluation**

For an initial distribution $\mu$ we define the performance of the policy

$$J_\mu(\pi) = \mathbb{E}_\mu^\pi\left[\sum_{t=0}^\infty \gamma^t R_t\right]$$

Other objects: value functions of a policy
- $V^\pi(s) = \mathbb{E}_s^\pi\left[\sum_{t=0}^\infty \gamma^t R_t\right]$  (vector)
- $Q^\pi(s,a) = \mathbb{E}_{s.a}^\pi\left[\sum_{t=0}^\infty \gamma^t R_t\right]$ (matrix)

Connection:
$$V^\pi(s) = \sum_{a\in \mathcal{A}} \pi(a|s)Q^\pi(s,a)$$

Bellman expectation: writting the definitions and run one step forward and use the Markov property. Bellman equations are basically linear algebra for $V$ and $Q$, but they involve a matrix inversion that does not scale and normally we don't want to do it. 

We can rewrite them as fix points of some operator $T^\pi$ and treat the problem from the point of view of contracting operators and fixed point iteration to obtain the solution. 

**Policy Optimization**

For a given state or state, action pair we define the optimal value functions:

$$V^*(s) = \argmax_\pi V^\pi(s) \qquad Q^*(s,a) = \argmax_\pi Q^\pi(s,a)$$
2
They can also be written with some operator $T^*$ which is also a contraction and $V^*$ and $Q^*$ are operator fixed points. The relationship among them is: 

Key point: work with the greedy policy ($\pi = greedy(Q)$) then $T^\pi Q = T^*  Q $

Key consequence: Solve $T^*$ then apply argmax to $Q^*$ and get the optimal policy. 

**Algorithms**

Do Banach iteration (assuming we can evaluate the operator), at some point we stop and we get the optimal policy applying greedy from the $Q$ function. 


**Model Based vs Model Free**

We want to be able to do the optimization just by 'playing' the game, not by knowing the rules by which it was developed. In other terms, we do not have access to $p$ directly, we can just sample from it. 

- Q-learning: At inner step of the loop we approximate it with one sample as SGD.
- Periodic Q-learning (DeepMind's Delayed Target Trick): Perform SGD with K steps while the Q function is frozen. 

## Key Concepts
-

## Algorithms / Methods Covered
-

## Intuitions & Insights
_Things that clicked, surprising connections, open questions._

## Papers & References
Saad and barton book?

-

## Questions I Still Have
-

