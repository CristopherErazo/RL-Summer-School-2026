# Basics of RL I — Shie Mannor, 03/06/26 (11:00 am - 12:30 am)

## Summary

**Learming - Model Based**

Estimate model from observations. We need to take care about concentration bounds in estimations of random variables averages. (How many samples we need for each $S,a$ )


Principle: Relatively unknown states are assumed to have larger reward. OFUL (Optimism in Face of Uncetainty).


**Online approx of mean**

...



**Q-learning Algorithm and Convergenge:**

- Init: $Q_0(s,a) = 0$ (arbitraty)
- Observe: $(s_t,a_t,r_t,s_{t+1})$
- Update: $Q_{t+1}(s_t,a_t) = Q_1(s_t,a_t) + \alpha_t(s_t,a_t) \Gamma_t$ where $\Gamma_t = r_t+\gamma \max_a Q_t(s_{t+1},a) - Q_t(s_t,a_t)$

$\alpha$ is the step size

If every $(s,a)$ is performed infinitely often and step size $\alpha$ satisfyes.

$$\sum_t \alpha_t = \infty \qquad \sum_t \alpha_t^2=O(1)$$

then $Q_t$ converges with prob 1 to $Q^*$.

Stochastic Approximation Proof Methodology: Partition the error into: contraction + zero mean noise.


**SARSA: on-policy**
- Init: $Q_0(s,a) = 0$ (arbitraty)
- Observe: $(s_t,a_t,r_t,s_{t+1},a_{t+1})$ where $a_{t+1} = \pi(s_{t+1};Q_t)$ output of the on-policy
- Update: $Q_{t+1}(s_t,a_t) = Q_1(s_t,a_t) + \alpha_t(s_t,a_t) \Gamma_t$ where $\Gamma_t = r_t+\gamma Q_t(s_{t+1},a_{t+1}) - Q_t(s_t,a_t)$

How to select the action $ \pi(s_{t+1};Q_t)?$

- $\epsilon$-greedy
- Softmax with decreasing temperature


**Monte-Carlo**

Directly learning from experience (model free). The simplest idea is to average the returns over trajectories, learn a value function. 

Pros: 
- Does not assume Markovian enviromnent.
- Extends naturally for function approximation.
- First visit is unbiased

Cons:
- Update at the end of complete epidose (wasteful?)
- Biased in every visit.


**Temporal Diferences**
Learn the value functions directly from experience, we don't want to learn the model. It uses incomplete episodes (before the end). Updates the estimates.

We can also do multiple look-ahead steps Temporal difference, computing the temporal difference as a exponential average 
$$(1-\lambda) \sum_n \lambda^{n-1} \Delta^{(n)}$$

The point for this is that we do not know the future rewards.

**TD($\lambda$) backward view**

Both views are equivalent at the end of the episode. The forward wiev is the theory justification, the backward view gives the online updates every time using incomplete information.

**Actor Critic**

Environment, actor, and critic that evaluates the actor. 
- Critic: Computes value function.
- Actor: Selects action, improves policy.





## Key Concepts
-

## Algorithms / Methods Covered
-

## Intuitions & Insights
_Things that clicked, surprising connections, open questions._

## Papers & References

https://sites.google.com/view/rlfoundations/home

## Questions I Still Have
-

