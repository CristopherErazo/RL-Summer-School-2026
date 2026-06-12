# RL & LLMs — Michal Valko, 11/06/26

## Summary

Algorithmic alignment. 

We want to use as little human supervision as possible. 

Deep learning works with tons of data and labeld. RL does not have a lot of data, you collect them from observations. 

**Current Pipeline**

* Random initialization
    * -> Pretraining
* Pretrained model (model reads all internet and predicts next word)
    * -> Mixed training
* Instruction following model (what are the numbers of R in strawberry? --> 3. We give them promt and expect full answer. Model ready to follow instructions. SInce they read all internet at this point models are racist, sexist, responds from what it has)
    * -> RLHF 
* Final model (Model learns how to be nice, how to reply correctly, how to write nicely, etc. This is hard to quantify, is better to do thumbs up/down in chatgpt responses)



**What data we have?**

* For MID training $(x,y)= $ (prompt, high quality answer)
* For RLHF $(x,y_w,y_l) = $ (promt, prefered answer, non-prefered answer)

**How to extract rewards**
Bradley - Terry model

$$P(y>y'|x) = \sigma(r(y|x) - r(y'|x))$$

* P is preference probability
* the reward function is learned
* sigmoid function 
* reward (ELO score)


**Things that are done**
Verifiable rewards math, code...



**Reward modelling**

$$\min_\theta L(\theta) = - \mathbb{E}_{(x,y_w,y_l)} \left[\log \sigma (r_\theta(y_w|x) - r_\theta(y_l|x)) \right]$$

**Naive RL HF**

* Given reward function
* Goal: find a policy (probabolity of next token predition LLM) that maximizes its expectation.

$$\max_\psi L_{RL}(\psi) = \mathbb{E}_{y\sim \pi_\psi(y|x)} \left[ r_{\hat\theta}(y|x)\right]$$

Could be done by any RL method: Proximal Policy Optimization (PPO), REINFORCE Leave one Out (RLOO), Shifted Q-learning (ShiQ). 

In practice you can train them in phases, have several rewards, verifiable, non-verifiable, etc...

**Pairwise preference over ELO score**

Instead of $(x,y)$ -> Reward model -> ELO $r$ -> $P(y>y'|x) = \sigma(r(y|x) - r(y'|x))$

we do directly: $(x,y,y')$ -> Preference Model -> $P(y>y'|x)$

Initialise it with a LLM prompted: “Given this prompt ‘x’ and two responses ‘y1’ and ‘y2’, which one do you prefer?” trained by SL with preference human data.

**From reward to policy**
Every method here optimizes the same KL-regularized reward:
$$\max_{\pi_\theta} \;\; \mathbb{E}_{x\sim D, y\sim\pi_\theta(\cdot|x)} r(x,y) - \beta D_{KL} \left(\pi_\theta(\cdot|x) |\pi_{ref}(\cdot |x)\right)$$

The reference is after the pretraining.

The optimal policy is known in closed form.


**Best response vs probability of winning**





## Papers & References
-

## Questions I Still Have
-

