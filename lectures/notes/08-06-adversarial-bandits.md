# Adversarial Bandits — Nicolò Cesa-Bianchi, 08/06/26


## Summary

Online learning and sequential decision making is a very big field. 

Data streams are ubiquitous: sensors, markets, user interactions

Classical statistical learning theory is useful when we have the data and we want to analize it.

Sequential learning/online learning incrementally adjust their models as new data arrive.

No notion of stationarity/no notion of LLN (contentration). We need to design algo that are fundamentelly different than what we would do in statistical learning. 

**Prediction with expert advice**

A sequential decision-making problem 
- $d$ actions
- Unknown deterministic assignment of losses to actions $l_t=(l_t(1),\cdots,l_t(d)) \in [0,1]^d$ for each time step $t=1,2,\cdots$

For all the times $t=1,2,\cdots$

- Player picks an action $I_t$ and incurs loss $l_t(I_t)$ 
- Player gets feedback information $l_t=(l_t(1),\cdots,l_t(d))$

The main notion we care about here is *regret*.

$$R_T = \mathbb{E} \left[ \sum_{t=1}^T l_t(I_t)\right] - \min_{i=1,\cdots,d}\sum_{t=1}^T l_t(i)$$

Linear problem of online linear optimization on the simplex. Achieved in one of the corners.






## Papers & References
-

## Questions I Still Have
-

