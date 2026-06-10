# Basics of RL II — Vincent François-Lavet, 04/06/26 (9:00 am - 10:30 am)

## Summary

What does it mean generalization in RL and how can we improve generalizations. There are many techniques that are different to what we do in Supervised Learning. 

**Motivation**

We use RL to achieve some sequence of actions learning task and we care about the practical aplications of it. 
In real scenarios we are constrained by the amount of trials we can do either because: 
- we can only access to simulations of real worl -reality gap- that can be overcomed only if we have a policy that generalizes.
- or in other cases we do have acces to the real world but is -limited data-

We can do both: build a simulator that is very close to the reality, but also we want to design a learning algorithm that generalizes well.

**Generalization in Supervised Learning**

A SL algo can be seen as a mapping from a dataset $D_{LS}$ of samples $(x,y) \sim (X,Y)$ into a prefictive model $f(x|D_{LS})$.

In general what we look is for a low bias - low overfitting model.

- Increase data -> Decrease variance
- Increase complexity of model -> Decrease bias

There is a bias variance trade off that dominate the generalization performance of these models.
- There is an error  indtroduced by the learning algorithm (bias)
- And an error due to limited data (parametric variance)

These bias - variance decomposition is performed in the superviced setting assuming a L2 loss, but in the RL setting the loss is the expected discounted returns. Nonetheless we can always try to divide the loss in one part that comes from the bias of the model and another that comes from the overfitting. 


**Bias and overfitting in RL**

In RL we consider the batch or *offline setting* where construct a policy $\pi_D$ based on a dataset $D\sim \mathcal{D}$ that is depending on the initial distribution of states, the environment and the *exploration* policy. 

$$\mathbb{E}_{\mathcal{D}} \left[V^{\pi^*}(x)-V^{\pi_D}(x)\right] = \left[V^{\pi^*}(x)-V^{\pi_{D_{\infty}}}(x)\right] - \mathbb{E}_{\mathcal{D}} \left[V^{\pi_{D_{\infty}}}(x)-V^{\pi_D}(x)\right]$$

where the firt term is the asymptotic bias and the second is the error due to the finite size data.


**How to improve genrealization?**

- adding an abstract representation that discards non essential features.
- modify the obkective function (e.g. reward shaping, tuning the training discount factor, etc.)
- with learning alorithms (type of function approximator and model free vs model based)
- improve the dataset (exploration vs exploitation)


## Papers & References
-

## Questions I Still Have
-

