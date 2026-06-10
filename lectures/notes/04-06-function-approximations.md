# Function Approximations — Vincent François-Lavet, 04/06/26

## Summary

In general RL will include:
- value funciton for how good is each state - state/action 
- direct representation of the policy
- model of the enviromnemt in conjunction with a planning algorithm 

**Recall**

$$V^\pi (s) = \mathbb{E} \left[\sum_{k=0}^\infty \gamma^k r_{t+k} | s_t = s , \pi\right]$$

$$Q^\pi (s,a) = \mathbb{E} \left[\sum_{k=0}^\infty \gamma^k r_{t+k} | s_t = s , a_t=a, \pi\right]$$

the optimal policy can be obtained as:

$$\pi^*(s) = \argmax_{a\in \mathcal{A}} Q^*(s,a)$$

With Bellman iteration we converge to $Q^*$.

$$Q(s_t,a_t) \leftarrow Q(s_t,a_t) + \alpha_t \left[ r_t+\gamma\max_{a'\in \mathcal{A}}Q_t(s_{t+1},a')-Q_t(s_t,a_t) \right]$$

**Limitations of tabular methods**

In large problems we are constrained by the course of dimensionality. For ex. in chess $\sim 10^{120}$ states or in go $\sim 10^{170}$ states

- Memory problems
- Compute time  
- No generalization in the limited data context.

For this reasons (and maybe others) we move to function approximators.

**Function Approximators**

$f: X \rightarrow Y$ parametrized by $\theta$, $y=f(x;\theta)$. In particular *neural networks* are a good function approximator because they are flexible and differentiable. 


To perform *gradient descent* we need a function $J(\theta)$ to be minimized that is differentiable with respect to $\theta$. 

**Q-learning with function approximations**

We represent value functions with a parametric function: 

$$Q(s,a;\theta) \approx Q(s,a)$$

The usual way to do it is to return, for a given $s$, all the $Q$ values for all the actions. 

$$s \rightarrow \text{Q-network } | \;\theta \rightarrow \{Q(s,a;\theta)\}_{a \in \mathcal{A}}$$

in this way later on is easier to compute softmax or symilar. This approach would not work if the actions space is huge and is not the architecture used for LLMs either. Another advantage is that even if you haven't visited a given state, the Q value will also get updated due to the other visited states.

The parameters $\theta$ are updated with gradient descent:

$$\theta \leftarrow \theta - \alpha \nabla_\theta\left( Q(s,a; \theta) - Y_k^Q \right)^2$$
where
$$Y_k^Q = r_k + \gamma \max_{a'\in \mathcal{A}} Q(s',a';\theta_k)$$





## Papers & References
-

## Questions I Still Have
-

