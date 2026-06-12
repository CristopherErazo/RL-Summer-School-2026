# Comprehensive Reinforcement Learning: From Fundamentals to Real-World Applications
## RL Summer School 2026 — Complete Course Summary

---

## Table of Contents
1. [Mathematical Foundations: MDPs & Dynamic Programming](#1-mathematical-foundations-mdps--dynamic-programming)
2. [Core RL Algorithms: Value-Based Learning](#2-core-rl-algorithms-value-based-learning)
3. [Generalization & Supervised Learning Principles](#3-generalization-in-rl)
4. [Function Approximation & Neural Networks](#4-function-approximation--neural-networks)
5. [Deep Reinforcement Learning](#5-deep-reinforcement-learning)
6. [Planning: Monte Carlo Tree Search](#6-planning-monte-carlo-tree-search-mcts)
7. [Bandits: Exploration-Exploitation Tradeoffs](#7-bandits-exploration-exploitation-tradeoff)
8. [Advanced Topics: Multi-Agent, Meta-Learning, Policy Optimization](#8-advanced-topics)
9. [Real-World Applications: LLMs, Health, Robotics](#9-real-world-applications)

---

## 1. Mathematical Foundations: MDPs & Dynamic Programming

### 1.1 Markov Decision Processes (MDPs)

**Definition**: An MDP is defined by the tuple $(𝒮, 𝒜, p, r, ν, γ)$ where:
- **State Space** $𝒮$: The set of all possible states
- **Action Space** $𝒜$: The set of available actions (may be state-dependent $𝒜_s$)
- **Transition Kernel** $p : 𝒮 × 𝒜 → 𝒮 × ℝ$: Defines dynamics $p(s', r | s, a)$
- **Reward Function** $r(s, a)$: Immediate reward signal
- **Initial Distribution** $ν$: Starting state distribution
- **Discount Factor** $γ ∈ [0,1)$: Weights immediate vs future rewards

**Key Property - The Markov Property**: 
The future state depends only on the current state and action, not on history:
$$P(s_{t+1}|s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1}|s_t, a_t)$$

This property enables efficient computation through dynamic programming.

### 1.2 Policies and Value Functions

**Policy**: A mapping from states to action probabilities:
$$π(a|s) = \mathbb{P}(A_t = a | S_t = s)$$

**Value Function** (expected cumulative discounted reward under policy $π$):
$$V^π(s) = 𝔼_s^π\left[\sum_{t=0}^∞ γ^t R_t\right]$$

**Action-Value Function** (Q-function):
$$Q^π(s,a) = 𝔼_{s,a}^π\left[\sum_{t=0}^∞ γ^t R_t\right]$$

**Relationship**:
$$V^π(s) = \sum_{a ∈ 𝒜} π(a|s)Q^π(s,a)$$

**Performance Objective**:
For initial distribution $μ$:
$$J_μ(π) = 𝔼_μ^π\left[\sum_{t=0}^∞ γ^t R_t\right]$$

### 1.3 Bellman Equations

**Policy Evaluation - Bellman Expectation Equation**:
$$V^π(s) = \sum_a π(a|s) \sum_{s',r} p(s',r|s,a)[r + γV^π(s')]$$
$$Q^π(s,a) = \sum_{s',r} p(s',r|s,a)[r + γ \sum_{a'} π(a'|s')Q^π(s',a')]$$

**Policy Optimization - Bellman Optimality Equation**:
$$V^*(s) = \max_a \sum_{s',r} p(s',r|s,a)[r + γV^*(s')]$$
$$Q^*(s,a) = \sum_{s',r} p(s',r|s,a)[r + γ \max_{a'} Q^*(s',a')]$$

**Greedy Policy Extraction**:
$$π^*(s) = \argmax_a Q^*(s,a)$$

### 1.4 Dynamic Programming

**Contraction Operators**: Bellman equations can be viewed as fixed-point equations:
- Operator $T^π$ for policy evaluation
- Operator $T^*$ for optimal value function
- Both are contraction operators (contraction factor $γ$)

**Value Iteration Algorithm**:
```
Initialize: V₀(s) = 0 for all s
For k = 0, 1, 2, ...
    V_{k+1}(s) ← max_a Σ_{s',r} p(s',r|s,a)[r + γV_k(s')]
```
Converges to $V^*$ as $k → ∞$

**Policy Iteration Algorithm**:
```
Initialize: π arbitrary
While π has not converged:
    1. Policy Evaluation: Compute V^π
    2. Policy Improvement: π' ← greedy(V^π)
    3. If π' = π, stop; else π ← π'
```

**Challenge**: 
- Requires knowledge of transition model $p$
- Matrix inversion scales poorly
- Infeasible for large state spaces
- Solution: Move to sampling-based methods

---

## 2. Core RL Algorithms: Value-Based Learning

### 2.1 Q-Learning: Model-Free Off-Policy Learning

**Motivation**: Learn optimal value function by sampling from environment without knowing $p$.

**Algorithm**:
```
Initialize: Q₀(s,a) = 0 for all (s,a)
Observe: (sₜ, aₜ, rₜ, s_{t+1})

Update rule:
Q_{t+1}(sₜ, aₜ) = Q_t(sₜ, aₜ) + αₜ(sₜ, aₜ) · Γₜ

where Γₜ = rₜ + γ max_{a'} Q_t(s_{t+1}, a') - Q_t(sₜ, aₜ)
```

**Key Features**:
- **Off-policy**: Can learn from data collected by any policy
- **Bootstrap**: Uses estimated Q-values for next state (biased but lower variance)
- **Temporal Difference (TD)**: Updates based on incomplete episodes

**Convergence Guarantee**:
If every state-action pair $(s,a)$ is visited infinitely often and step size $α_t$ satisfies:
$$\sum_t α_t = ∞ \quad \text{and} \quad \sum_t α_t^2 < ∞$$

Then $Q_t(s,a) → Q^*(s,a)$ with probability 1.

**Convergence Proof Technique - Stochastic Approximation**:
Decompose error into two parts:
1. **Contraction**: Bellman operator contracts the space
2. **Zero-Mean Noise**: Updates are unbiased (expectation is the contracted value)

### 2.2 SARSA: On-Policy Temporal Difference

**Algorithm**:
```
Observe: (sₜ, aₜ, rₜ, s_{t+1}, a_{t+1}) where a_{t+1} ~ π(s_{t+1})

Update rule:
Q_{t+1}(sₜ, aₜ) = Q_t(sₜ, aₜ) + αₜ · [rₜ + γQ_t(s_{t+1}, a_{t+1}) - Q_t(sₜ, aₜ)]
```

**Key Differences from Q-Learning**:
- **On-policy**: Learns value of the policy being followed
- Uses actual next action from policy, not maximum
- More conservative than Q-learning (follows off-policy distribution)
- Converges even with $ε$-greedy exploration

**Policy Selection Strategies**:
- **ε-greedy**: With probability ε, pick random action; else pick greedy action
- **Softmax**: Select action proportional to exp(Q(s,a)/τ) where τ is temperature

### 2.3 Monte Carlo Methods

**Algorithm**:
1. Generate complete episode: $(s_0, a_0, r_0, s_1, a_1, r_1, ..., s_H)$
2. Compute return: $G_t = \sum_{k=0}^{H-t} γ^k r_{t+k}$
3. Update: $V(s) ← V(s) + α[G_t - V(s)]$

**Advantages**:
- No assumption of Markov environment (more general)
- First-visit is unbiased estimator of $V^π$
- Naturally extends to function approximation

**Disadvantages**:
- Updates only at episode end (wasteful)
- High variance (depends on full trajectory)
- Biased if using every-visit average

### 2.4 Temporal Difference and TD(λ)

**TD(0)** - One-step lookahead:
$$V(s) ← V(s) + α[r + γV(s') - V(s)]$$

**n-step TD** - n-step lookahead:
$$V(s) ← V(s) + α[G_t^{(n)} - V(s)]$$
where $G_t^{(n)} = \sum_{k=0}^{n-1} γ^k r_{t+k} + γ^n V(s_{t+n})$

**TD(λ)** - Exponential average of n-step returns:
$$(1-λ) \sum_{n=1}^∞ λ^{n-1} G_t^{(n)}$$

**Backward View** (equivalent):
- Maintain eligibility traces $e_t(s)$ for each state
- Updates propagate backward through trajectory
- Online computation with incomplete episodes

**Why TD(λ)?**
- λ=0: Pure TD(0), low bias, high variance
- λ=1: Pure Monte Carlo, high bias, low variance
- 0<λ<1: Balance between bias and variance

### 2.5 Actor-Critic Architecture

**Components**:
1. **Critic**: Estimates value function $V^π(s)$ or $Q^π(s,a)$
2. **Actor**: Selects and improves policy π

**Algorithm**:
```
1. Observe state s
2. Actor selects action a ~ π(·|s)
3. Critic evaluates: δ = r + γV(s') - V(s)
4. Both update using TD error δ:
   - Actor: improve policy toward higher rewards
   - Critic: update value estimate
```

**Advantages**:
- Reduces variance of policy gradient (critic as baseline)
- Can use continuous action spaces
- Good for exploration-exploitation

---

## 3. Generalization in RL

### 3.1 Supervised Learning Perspective

**Goal**: Train model on training distribution that performs well on test distribution.

**Bias-Variance Decomposition** (L2 loss):
$$\text{Error} = \text{Bias}^2 + \text{Variance}$$

**Bias**: 
- Error introduced by learning algorithm
- Increases with model complexity
- Related to approximation error of function class

**Variance**:
- Error due to limited data
- Decreases with more training data
- Related to sample complexity

**Trade-off**:
- More complex model → Lower bias, higher variance
- More data → Lower variance
- Optimal model complexity depends on data availability

### 3.2 Generalization in RL

**Batch/Offline RL Setting**:
- Construct policy $π_D$ from dataset $D ~ 𝒟$
- $𝒟$ depends on: initial distribution, environment, exploration policy

**Offline Policy Evaluation**:
$$𝔼_𝒟[V^{π^*}(x) - V^{π_D}(x)] = \underbrace{[V^{π^*}(x) - V^{π_{D_∞}}(x)]}_{\text{Asymptotic Bias}} - \underbrace{𝔼_𝒟[V^{π_{D_∞}}(x) - V^{π_D}(x)]}_{\text{Estimation Error}}$$

Where $π_{D_∞}$ is the policy trained on infinite data.

**Key Differences from Supervised Learning**:
1. **Non-stationary Targets**: Q-learning targets depend on current network weights
2. **Data Non-IID**: Samples correlated due to sequential nature
3. **Distribution Shift**: Policy changes → state distribution changes

### 3.3 Improving Generalization in RL

**Strategies**:

1. **Abstract Representations**:
   - Learn features that discard non-essential information
   - Reduce effective state dimension

2. **Modify Objective Function**:
   - Reward shaping: Add auxiliary rewards
   - Adjust discount factor: Tune time horizon emphasis
   - Constrain policy divergence

3. **Learning Algorithm Design**:
   - Choose appropriate function approximator
   - Model-based vs model-free approaches
   - Regularization techniques

4. **Dataset Improvement**:
   - Better exploration strategy
   - Increase exploration-exploitation balance
   - Collect more diverse data

---

## 4. Function Approximation & Neural Networks

### 4.1 Limitations of Tabular Methods

**Curse of Dimensionality**:
- Chess: ~10^120 possible states
- Go: ~10^170 possible states
- Atari images: 10^18 possible observations

**Problems**:
- Memory: Cannot store all state-action pairs
- Computation: Cannot visit all states
- Generalization: No learning across similar states

### 4.2 Function Approximation Framework

**Parametric Representation**:
$$Q(s,a;θ) ≈ Q^*(s,a)$$
$$V(s;θ) ≈ V^*(s)$$

**Architecture for Discrete Actions**:
Input: State $s$
Output: Q-values for all actions
$$s → \text{Neural Network}_θ → \{Q(s,a;θ)\}_{a∈𝒜}$$

**Advantages**:
- Compact representation
- Generalization to unseen states
- Differentiable for gradient descent

### 4.3 Q-Learning with Function Approximation

**Update Rule**:
$$θ ← θ - α ∇_θ (Q(s,a;θ) - Y_k^Q)^2$$

where

$$Y_k^Q = r_k + γ \max_{a'∈𝒜} Q(s',a';θ_k)$$

**Issues with Naive Approach**:
1. **Moving Target**: θ appears in both prediction and target
2. **Correlation**: Sequential samples are correlated
3. **Instability**: Can cause divergence or oscillation

---

## 5. Deep Reinforcement Learning

### 5.1 The Deep RL Problem

**Core Challenge**: Combine function approximation (deep neural networks) with RL learning dynamics.

**Three RL Paradigms**:
1. **Policy-based**: Learn π directly → Execute it
2. **Value-based**: Learn V(s) or Q(s,a) → Extract policy by inspection
3. **Model-based**: Learn environment model → Infer policy by planning

**Deep RL**: Uses deep neural networks as function approximators in any/all paradigms.

### 5.2 All Problems as Functions

**Universal Representation**:
- **Policy**: $π : 𝒮 → P(𝒜)$ (distribution over actions)
- **Value**: $V : 𝒮 → ℝ$ (scalar value)
- **Model**: $m : 𝒮 × 𝒜 → 𝒮 × ℝ$ (next state and reward)

**Deep RL Approach**: Parameterize each as large neural network with parameters $θ$ trained via gradient descent.

### 5.3 Deep Prediction Methods

**Deep Monte Carlo Prediction**:
Learn state values $v_θ(s)$ of fixed policy $π$.

Loss function:
$$L = (G_t - v_θ(s_t))^2$$

Update:
$$Δθ = -∇_θL = (G_t - v_θ(s_t))∇_θ v_θ(s_t)$$

**Issue**: High variance (depends on full episode return)

**Deep TD(0) Prediction**:
$$Δθ = (R_t + γv_θ(s_{t+1}) - v_θ(s_t))∇_θ v_θ(s_t)$$

**Advantage**: Lower variance using bootstrapping

### 5.4 Deep Q-Learning

**Update Rule**:
$$Δθ = (R_t + γ \max_{a'} q_θ(s_{t+1},a') - q_θ(s_t,a_t))∇_θ q_θ(s_t,a_t)$$

### 5.5 Experience Replay

**Motivation**: Address correlation and non-stationarity issues.

**Algorithm**:
1. Collect transitions $(s_t, a_t, r_t, s_{t+1})$ in replay buffer
2. Sample minibatch uniformly at random from buffer
3. Compute loss on minibatch
4. Update network parameters

**Benefits**:
- Breaks temporal correlation
- Larger buffer → less correlated samples
- Reuse data (sample efficiency)
- Can prioritize novel/important transitions

### 5.6 Target Networks

**Problem with Moving Target**:
- Both $q_θ(s,a)$ and $\max_{a'} q_θ(s',a')$ use same network
- Creates feedback loop → instability

**Solution - Target Network**:
- Maintain two networks: online $θ$ and target $θ^-$
- Use target network for bootstrap targets
- Update target network every K steps: $θ^- ← θ$

**Stabilization Effect**:
- Breaks feedback loop
- Reduces variance of targets
- Enables stable learning

### 5.7 Generalization in Deep RL

**Sources of Generalization Error**:
1. Function approximation: NN smooth, target non-smooth (leakage)
2. Bootstrapping: Self-generated targets
3. Off-policy learning: Data distribution != policy distribution

**Deadly Triad**:
Function Approximation + Bootstrapping + Off-Policy = Divergence Risk

**Common Remedies**:
- Target networks (stability)
- Double Q-learning (overestimation correction)
- Dueling networks (separate value and advantage)
- Prioritized experience replay

### 5.8 State Representations

**Rich Representation Learning**:
1. **Pseudo-Rewards**: Auxiliary losses based on representations
2. **Auxiliary Tasks**: 
   - Immediate reward prediction
   - Next state prediction
   - Contrastive learning
3. **Multi-Scale Learning**: Different time horizons
4. **World Models**: Predict dynamics (VAE, RNN, Transformer)

### 5.9 Distributional Deep RL

**Beyond Expected Value**:
- Predict full reward distribution, not just mean
- Capture uncertainty
- Better exploration signals
- More stable learning

**Representation**: Q-network outputs distribution parameters instead of point estimates.

---

## 6. Planning: Monte Carlo Tree Search (MCTS)

### 6.1 Planning Problem

**Question**: Given current state, which action should agent take to maximize future rewards?

**Challenge**: Cannot exhaustively search all possible futures.

### 6.2 Game Trees for Planning

**Representation**:
- Root node: current state
- Child nodes: states reachable by one action
- Tree expands by taking actions

**Two-Player View**:
- Agent: chooses actions (minimizes cost or maximizes reward)
- Environment: stochastically samples next state

**Node Statistics**:
- Parent pointer (for backpropagation)
- Cumulative reward collected along path
- Visit count

### 6.3 Monte Carlo Tree Search Algorithm

**Core Idea**: Selectively expand promising branches using random sampling.

**Four Phases**:

1. **Selection**: Navigate tree using UCB-like strategy
2. **Expansion**: Add new child node
3. **Simulation**: Random playout from new node
4. **Backpropagation**: Update statistics up to root

**Trade-off**: Avoid exhaustive search while focusing on promising actions.

**Applications**: 
- AlphaGo (combined with deep neural networks)
- Computer game AI
- Planning under uncertainty

---

## 7. Bandits: Exploration-Exploitation Tradeoff

### 7.1 The Bandit Problem

**Simplest RL Setting**: Single state, k actions (bandits).

**Formal Model**:
- k actions with unknown means $μ ∈ [0,1]^k$
- Known time horizon n
- Each round: choose action $A_t$, receive reward $R_t ~ B(μ_{A_t})$
- **Exploration-Exploitation Dilemma**: 
  - Exploit: Play known good actions
  - Explore: Try unknown actions to learn better option

**Real-World Analogs**:
- Coffee shop selection: known vs. new
- Medical trials: proven treatment vs. experimental
- Research: established methods vs. novel directions

### 7.2 Performance Metrics

**Cumulative Regret**:
$$\text{Reg}_n = 𝔼\sum_{t=1}^n (μ_1 - μ_{A_t}) = 𝔼\sum_{t=1}^n Δ_{A_t}$$

(Assume action 1 is optimal)

**Interpretation**: 
- Total opportunity loss from not always playing best action
- Lower regret = better algorithm

### 7.3 Confidence Bound Approach

**Core Idea - Optimism Under Uncertainty**:
- Estimate mean and uncertainty for each bandit
- Play action with highest upper confidence bound
- Optimistic principle: act as if in best plausible scenario

**Empirical Means**:
$$\hat{μ}_a(t) = \frac{\text{sum of rewards from } a}{\text{number of pulls of } a}$$

**Hoeffding's Inequality**:
For $X_1, ..., X_n$ iid random variables in [0,1] with mean $μ$:
$$\mathbb{P}(|\hat{μ} - μ| ≥ \sqrt{\frac{\ln(2/δ)}{2n}}) ≤ δ$$

**Upper Confidence Bound (UCB)**:
$$\text{UCB}_a(t) = \hat{μ}_a(t) + \sqrt{\frac{\ln(2/δ)}{2n_a(t)}}$$

where $n_a(t)$ is number of times action $a$ pulled.

**Algorithm**:
```
For each round t:
    For each action a:
        Compute UCB_a(t)
    A_t ← argmax_a UCB_a(t)
    Pull A_t, observe R_t
    Update n_{A_t}(t) and reward sum
```

### 7.4 Key RL Insight

**Non-IID Data**:
- In classical statistics: samples are IID
- In bandits: future actions depend on past observations
- Solution: Imagine drawing all rewards upfront (iid), then selectively revealing them

**Two Perspectives**:
- $\hat{μ}_a(t)$: Mean observed so far (non-IID, depends on exploration policy)
- $\hat{μ}_{at}$: Mean if we replayed bandit a, t times (IID, counterfactual)

### 7.5 Policy Gradient for Bandits

**Extension**: Apply gradient-based methods to bandit problem.

**Connection to RL**: Bandits are 1-state MDPs; most RL algorithms reduce to bandit algorithms in this setting.

---

## 8. Advanced Topics

### 8.1 Adversarial Bandits

**Problem Shift**: Rewards are not random; adversary chooses them adversarially.

**Setting**:
- d actions
- Each round, adversary assigns loss $l_t(i) ∈ [0,1]$ to each action
- Player chooses action $I_t$, incurs loss $l_t(I_t)$
- Player observes all losses $l_t = (l_t(1), ..., l_t(d))$

**No Stationarity Assumption**: Losses can change arbitrarily.

**Regret Definition**:
$$\text{Reg}_T = 𝔼\left[\sum_{t=1}^T l_t(I_t)\right] - \min_{i} \sum_{t=1}^T l_t(i)$$

Compares to best fixed action in hindsight.

**Prediction with Expert Advice**:
- Multiple experts (or actions)
- Combine expert predictions
- Hedge strategy: maintain weights, downweight poor performers
- Achieve regret = $O(\sqrt{T \ln d})$

### 8.2 Performance Difference Lemma and Policy Optimization

**Goal**: Foundation for policy-based RL methods.

**Starting Point - Alternative Characterization**:

Instead of Bellman equations, use **occupancy measures**.

**Occupancy Measure**:
$$ρ^π(s,a) = (1-γ) 𝔼\left[\sum_{t=0}^∞ γ^t \mathbb{1}_{S_t=s, A_t=a} | π\right]$$

**Interpretation**: Discounted frequency of state-action pair (s,a) under policy π.

**Policy Performance via Occupancy**:
$$V(π) = \sum_{s,a} ρ^π(s,a) r(s,a)$$

**Performance Difference Lemma**:
$$V(π') - V(π) = \frac{1}{1-γ} 𝔼_{ρ^{π'}}\left[Q^π(s,a) - V^π(s)\right]$$

Relates change in value to advantage function under new occupancy measure.

**Policy Optimization Methods**:
- Maximize over occupancy measures
- Recover policy from optimal occupancy
- Theoretical foundation for policy gradient methods

### 8.3 Meta-Reinforcement Learning

**Motivation**:
- Robots must adapt to novel situations quickly
- Cannot rely on slow trial-and-error learning
- Need to leverage prior experience

**Setting**:
- Multi-task RL: training tasks $M_1, ..., M_N$
- Each task is MDP $M_i = \{P_i(s'|s,a), r_i(s,a)\}$
- Task distribution: $P(M)$

**Goal**:
$$\max_π 𝔼_{M ~ P(M)} V_M(π)$$

**Key Insight**:
Agent doesn't know which MDP it's in → **Partially Observable MDP (POMDP)**.

**POMDP Formulation**:
- States s (hidden)
- Actions a (observable)
- Observations o (from state)
- History: $h_t = (o_0, a_0, r_0, ..., o_t)$
- Policy: $π(a|h_t)$ (history-dependent, non-Markovian)

**Information States**:
- Summary statistics of history: $z_t = f_t(h_t)$
- Enable dynamic programming
- Examples:
  - Full history: $z_t = h_t$
  - Belief state: $z_t = P(s_t | h_t)$ (Bayesian belief)

**World Models for Meta-RL**:
Learn compact representation of history:
- **LSTMs**: Sequential processing
- **Transformers**: Attention-based
- **VAE**: Latent representation
- **Classifiers**: Auxiliary tasks

Approaches: Self-supervised learning on trajectories.

**Practical Framework**:
1. Access to N training tasks (simulators)
2. Learn policy that works across tasks
3. Minimize regret: $R(π) = L(π_{BO}) - L(π)$

### 8.4 Multi-Agent Reinforcement Learning

**Key Difference from Single-Agent**:
- Multiple agents with potentially conflicting objectives
- Each agent's action affects others' rewards
- No single "optimal policy" definition

**Normal Form Games**:
- Simplest multi-agent setting
- No states, no transitions
- Rewards fully specified: $R_i ∈ ℝ^{|A_1| × |A_2| × ...}$
- One-shot interaction

**Zero-Sum Games** (2-player):
- Pure adversarial: $R_1 = -R_2$
- One player's gain is another's loss

**Nash Equilibrium**:
- Policy profile $(π_1^*, ..., π_n^*)$ where no agent can improve by unilateral deviation
- Multi-agent analog of Bellman optimality
- Fixed point where all agents best-respond to each other

$$π_i^*(a_i|s) = \text{best response to } π_j(a_j|s) \text{ for all } j ≠ i$$

**Computational Challenges**:
Even with full information, finding Nash equilibrium is hard.

**Real Complexities**:
- Information asymmetry
- Partial observability
- Non-stationarity (other agents learning)
- Communication/coordination

---

## 9. Real-World Applications

### 9.1 Real-World RL Challenges

**Fundamental Differences from Simulation**:

**Simulation Loop**:
```
Policy → Action → Simulator 
(full state, deterministic, reset) 
→ State + Reward (millions of cheap steps)
```

**Robot Loop**:
```
Policy → Torque/Target → Robot + World
(contact, wear, real-time)
→ Noisy Partial Observations (seconds each, human reset)
```

**Key Challenges**:

1. **Transition Model**:
   - Simulation rarely accurate
   - Physics approximations, contact dynamics
   - Real-time constraints
   - Wear and degradation

2. **Reward Function**:
   - Hard to specify precisely
   - Often design itself is the work
   - Contact tasks: most effort goes here
   - Multi-objective: competing goals

3. **Policy Deployment**:
   - Safe exploration impossible (dangerous to try wrong actions)
   - Trial-and-error not feasible
   - Need to leverage simulation while reducing reality gap
   - Require robust, reliable policies

**Lessons**:
- Benchmarks are nice but not the real goal
- Full deployment pipeline is complex
- Cannot optimize single metric in isolation

### 9.2 Reinforcement Learning from Human Feedback (RLHF) for LLMs

**Problem**: How to align large language models with human preferences?

**Challenge**: 
- Reward is "being helpful, harmless, honest"
- Hard to specify formally
- Deep learning works with massive labeled data
- RL has less data but can learn from indirect signals

**Current Pipeline**:

1. **Pretraining**:
   - Random initialization
   - Massive self-supervised learning
   - Predict next token on internet data
   - Result: Model reads all internet, predicts words

2. **Instruction Tuning** (Supervised Fine-Tuning):
   - Dataset: $(x, y) = (\text{prompt}, \text{high-quality answer})$
   - Supervised learning: model learns to follow instructions
   - Result: Instruction-following model (but reflects internet biases)

3. **RLHF** (Reinforcement Learning from Human Feedback):
   - Dataset: $(x, y_w, y_l) = (\text{prompt}, \text{preferred answer}, \text{non-preferred})$
   - Learn to rank: preferred response > non-preferred
   - RL objective: maximize preference satisfaction

**Reward Modeling**:

**Bradley-Terry Model**:
$$P(y > y'|x) = σ(r(y|x) - r(y'|x))$$

- $P$: probability preferred response is better
- $r$: learned reward function (like ELO score)
- $σ$: sigmoid (maps difference to probability)

**Reward Learning Loss**:
$$\min_θ L(θ) = -𝔼_{(x,y_w,y_l)}\left[\ln σ(r_θ(y_w|x) - r_θ(y_l|x))\right]$$

**Naive RL HF**:
```
1. Train reward function r_θ from preference data
2. Use reward to optimize LLM policy:
   
   max_ψ L_{RL}(ψ) = 𝔼_{y ~ π_ψ(y|x)} [r_θ̂(y|x)]
```

**Practical Algorithms**:
- PPO (Proximal Policy Optimization)
- RLOO (REINFORCE Leave-One-Out)
- ShiQ (Shifted Q-learning)

**KL-Regularized Objective**:
$$\max_{π_θ} 𝔼_{x ~ D, y ~ π_θ(·|x)}\left[r(x,y) - β D_{KL}(π_θ(·|x) || π_{ref}(·|x))\right]$$

- $r$: learned reward signal
- $D_{KL}$: divergence from reference (pretrained) policy
- $β$: regularization weight

**Intuition**: Stay close to pretrained model while improving on reward signal.

**Why This Works**:
- Reference policy acts as regularization
- Prevents collapse to gaming the reward model
- Maintains language quality while improving alignment
- Optimal policy: closed-form solution known

**Practical Considerations**:
- Verifiable rewards (math, code) easier to optimize
- Non-verifiable rewards (style, tone) harder
- Multiple reward signals possible (ensemble)
- Training in phases with different reward sets

### 9.3 Preference Models

**Direct Preference Optimization**:

Instead of:
```
Data → Reward Model → ELO Scores → Policy Optimization
```

Do directly:
```
Data → Preference Model → Probability of Preference → Policy Optimization
```

**Initialization**:
- Start with LLM prompted: "Which response do you prefer?"
- Train on preference pairs using SL

**Advantage**: 
- More direct signal
- Fewer components
- Less reward hacking

### 9.4 RL for Healthcare

**Unique Challenges**:

1. **Partial Observability**:
   - Medical state incompletely observed
   - Observations are noisy or incomplete
   - Makes process non-Markovian

2. **Confounding**:
   - Correlated unmeasured variables
   - Treatment decisions affected by hidden state
   - Can lead to wrong causal conclusions

**Mitigation Strategies**:

1. **Causal RL**:
   - Learn causal graph from data
   - Use instrumental variables
   - Explicit causal inference

2. **Sensitivity Analysis**:
   - Quantify how much confounding reverses conclusions
   - Robustness checks

3. **Negative Controls**:
   - Outcomes known to be unaffected by treatment
   - If causal effect found on negative control → error
   - Ensures validity

**Distribution Shift**:
- Hospital A data different from Hospital B
- Different populations, treatment protocols
- External validity challenge
- Need domain adaptation techniques

### 9.5 Robotics Applications

**Fundamental Gap: Gym vs. Real Robot**

**Simulation (Gym)**:
```
s ← (discrete/continuous state, complete)
a ← action applied instantly
s', r ← returned immediately (deterministic or known distribution)
reset() ← possible, free
Timing: Can do millions of steps
```

**Real Robot**:
```
s ← (images, proprioception, force-torque: partial, delayed)
a ← command to controller (not direct joint control)
    → Robot executes in real-time with contact dynamics
s', r ← noisy, partial observations after seconds
reset ← human must physically reset (expensive, slow)
Timing: Seconds per step, human attention required
```

**Key Challenges**:

1. **Partial Observations**:
   - Cannot directly observe full state
   - Combine images, sensors, proprioception
   - Temporal integration needed

2. **Real Dynamics**:
   - Contact dynamics (friction, stiction, bouncing)
   - Actuator constraints (torque limits, delays)
   - Environmental variation (temperature, wear)
   - Wear and degradation over time

3. **Action Representation**:
   - Command is to controller, not direct forces
   - Different levels of abstraction
   - Hierarchical control often needed

4. **Reward Design**:
   - **Most important and challenging**
   - For contact tasks: contact rewards dominate
   - Multi-objective: stability, efficiency, safety
   - Often requires careful engineering

**Design Considerations**:
- Theory/algorithms relatively solved
- Real challenge: bridging theory-embodiment gap
- Sim-to-real transfer
- Safe exploration strategies

---

## 10. Integrative Framework

### 10.1 Taxonomy of RL Approaches

```
RL Algorithms
├── Value-Based
│   ├── Tabular: Q-learning, SARSA
│   ├── Approximate: Q-learning + function approximation
│   └── Deep: DQN, Double DQN, Dueling DQN
├── Policy-Based
│   ├── Policy Gradient: REINFORCE, Actor-Critic
│   ├── Trust Region: TRPO, PPO
│   └── Maximum Entropy: SAC
└── Model-Based
    ├── Planning: MCTS, Dyna
    ├── World Models: Model learning + planning
    └── Imagination-Augmented: Latent planning
```

### 10.2 When to Use What

**Value-Based Methods**:
- Discrete action spaces
- Off-policy learning important (sample efficiency)
- Convergence guarantees useful
- Example: Game playing (Atari, Go)

**Policy-Based Methods**:
- Continuous action spaces
- Want to learn stochastic policies (exploration)
- On-policy acceptable
- Example: Robotics, control tasks

**Model-Based Methods**:
- Data efficiency critical
- Long horizons need planning
- Transition model learnable
- Example: Planning, hierarchical control

**Hybrid Approaches**:
- Often best in practice
- Actor-critic: policy + value
- World models + planning: imagination

### 10.3 Key Design Decisions

1. **On-policy vs. Off-policy**:
   - Off-policy: More sample efficient, harder to learn
   - On-policy: Simpler, requires more interactions

2. **Bootstrapping vs. Monte Carlo**:
   - Bootstrap: Lower variance, biased
   - MC: Unbiased, high variance
   - TD(λ): Balance via eligibility traces

3. **Function Approximation**:
   - Linear: Stable, interpretable, limited capacity
   - Neural networks: Flexible, but instability issues
   - Need target networks, replay, regularization

4. **Exploration Strategy**:
   - Deterministic: ε-greedy, softmax
   - Stochastic: Policy entropy, intrinsic motivation
   - Active learning: Information-seeking

---

## 11. Key Theoretical Insights

### 11.1 Convergence Guarantees

**Conditions for Convergence**:
1. Every state-action pair visited infinitely
2. Step size decreases appropriately: $\sum α_t = ∞$, $\sum α_t^2 < ∞$
3. Bellman operator is contraction (guaranteed for value functions)
4. Function approximation: additional conditions needed

**No Convergence**:
- Deadly triad: Function approximation + Bootstrapping + Off-policy

### 11.2 Sample Complexity

**PAC-Bounds** (Probably Approximately Correct):
- Quantify how many samples needed for ε-optimal policy
- Depends on: state space, action space, horizon, accuracy ε, confidence δ
- Can be exponential in state space dimension → need approximation

**Regret Bounds**:
- Cumulative regret over time
- Stochastic bandits: $O(\ln T)$ regret achievable
- Adversarial bandits: $O(\sqrt{T})$ regret achievable
- MDP structure can enable better bounds

### 11.3 Exploration-Exploitation Tradeoff

**Information Gain vs. Immediate Reward**:
- Exploration: try uncertain actions to learn
- Exploitation: play best known action

**Optimism Under Uncertainty**:
- UCB-type algorithms
- Upper confidence bounds on values
- Principled balance

**Thompson Sampling**:
- Maintain posterior distribution over rewards
- Sample from posterior
- Play sampled best action
- Automatically balances exploration/exploitation

### 11.4 Causal Inference in RL

**Challenge**: Offline learning with non-random data
- Observational data affected by previous policies
- Confounding: unmeasured variables affect treatment and outcome

**Solutions**:
- Causal graphs: learn structure
- Instrumental variables: use proxy variables
- Sensitivity analysis: bounds under confounding
- Negative controls: sanity checks

---

## 12. Summary: From Theory to Practice

### 12.1 Complete RL Pipeline

1. **Problem Formulation**:
   - Define state space, action space, reward function
   - Determine whether model available
   - Assess data constraints

2. **Algorithm Selection**:
   - Choose value-based, policy-based, or model-based
   - Consider on-policy vs. off-policy
   - Decide on function approximation

3. **Implementation**:
   - Neural network architecture
   - Hyperparameter tuning
   - Exploration strategy

4. **Stabilization Techniques**:
   - Target networks
   - Experience replay
   - Regularization
   - Reward scaling

5. **Evaluation**:
   - Sample efficiency
   - Asymptotic performance
   - Generalization to new tasks/domains
   - Real-world viability

### 12.2 Common Pitfalls

1. **Non-IID Samples**: 
   - Use experience replay
   - Break temporal correlation

2. **Non-Stationary Targets**:
   - Use target networks
   - Update periodically

3. **Exploration Failure**:
   - Adjust exploration strategy
   - Use curiosity/intrinsic motivation
   - Consider count-based exploration

4. **Overfitting to Reward**:
   - Validate on holdout environment
   - Use multiple reward signals
   - Add constraints/regularization

5. **Reality Gap**:
   - Domain randomization
   - Conservative policy optimization
   - Robust reward design

### 12.3 Current Frontiers

1. **Sample Efficiency**:
   - Data-efficient learning algorithms
   - Transfer learning across tasks
   - Self-supervised pre-training

2. **Robustness**:
   - Adversarial robustness
   - Distribution shift handling
   - Conservative policies

3. **Alignment**:
   - Learning from human feedback
   - Value learning
   - Corrigibility and interpretability

4. **Scalability**:
   - Large state/action spaces
   - Hierarchical approaches
   - Curriculum learning

5. **Real-World Deployment**:
   - Safe exploration
   - Efficient resets
   - Robust to mismatch
   - Interpretable decisions

---

## Bibliography & Key Papers

**Foundations**:
- Sutton & Barto (2018) - "Reinforcement Learning: An Introduction"
- Bellman (1957) - "Dynamic Programming"

**Core Algorithms**:
- Watkins (1989) - Q-learning
- Konda & Tsitsiklis (2000) - Actor-Critic

**Deep RL**:
- Mnih et al. (2015) - DQN (Atari)
- Hessel et al. - Rainbow (Multiple improvements)

**Policy Optimization**:
- Schulman et al. (2015) - Trust Region Policy Optimization (TRPO)
- Schulman et al. (2017) - Proximal Policy Optimization (PPO)

**Exploration**:
- Lattimore & Szepesvári (2020) - "Bandit Algorithms"
- Cesa-Bianchi & Lugosi (2006) - "Prediction, Learning, and Games"

**LLM Alignment**:
- Ouyang et al. (2022) - RLHF for InstructGPT
- Christiano et al. (2017) - Learning from human feedback

**Multi-Agent**:
- Shapley (1953) - Nash Equilibrium
- Littman (1994) - Multi-agent learning

**Practical Applications**:
- AlphaGo papers (Silver et al.)
- Robotics: Levine et al. on offline RL
- Healthcare: Liu et al. on causal RL

---

## Appendix: Key Equations Summary

| Concept | Equation |
|---------|----------|
| Value Function | $V^π(s) = 𝔼_s^π[\sum_{t=0}^∞ γ^t R_t]$ |
| Q-Function | $Q^π(s,a) = 𝔼_{s,a}^π[\sum_{t=0}^∞ γ^t R_t]$ |
| Bellman Expectation | $V^π(s) = \sum_a π(a\|s)\sum_{s',r} p(s',r\|s,a)[r + γV^π(s')]$ |
| Bellman Optimality | $V^*(s) = \max_a \sum_{s',r} p(s',r\|s,a)[r + γV^*(s')]$ |
| Q-Learning Update | $Q_{t+1}(s,a) = Q_t(s,a) + α[r_t + γ\max_{a'} Q_t(s',a') - Q_t(s,a)]$ |
| TD Error | $δ_t = r_t + γV(s_{t+1}) - V(s_t)$ |
| Policy Gradient | $∇J(θ) = 𝔼[∇ \ln π(a\|s;θ) Q(s,a)]$ |
| Deep Q-Loss | $L(θ) = (r + γ\max_{a'} q_θ(s',a') - q_θ(s,a))^2$ |
| UCB Bonus | $\text{UCB}_a = \hat{μ}_a + \sqrt{\frac{\ln(2/δ)}{2n_a}}$ |
| Bradley-Terry | $P(y > y'\|x) = σ(r(y\|x) - r(y'\|x))$ |

---

**Document Summary**: This comprehensive guide traces the evolution of RL from mathematical foundations through cutting-edge applications. Starting with MDPs and dynamic programming, it builds through value-based and policy-based methods, deep RL, and specialized topics like bandits and multi-agent learning. The latter chapters address real-world challenges in LLMs, healthcare, and robotics, highlighting the practical considerations often overlooked in academic treatments. The material emphasizes both theory and practice, providing equations, algorithms, and intuitions for understanding when and how to apply different approaches.
