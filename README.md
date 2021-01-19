# Model-Based Method - Dynamic Programming

## Introduction
In this session, you'll learn the most basic strategy in reinforcement learning (**Dynamic Programming**) to compute optimal policies **given the model of the environment**. You'll learn that there are two basic steps for solving any RL problem.
* **Policy evaluation** refers to the iterative computation of the value functions for a given policy.
* **Policy improvement** refers to finding an improved policy given the value function for an existing policy.

You'll learn about two basic approaches to solve an RL problem:
* **Policy Iteration** and
* **Value Iteration**

### Dynamic Programming
**Dynamic Programming (DP)** is a method of solving complex problems by breaking them into sub-problems. It solves the sub-problems and stores the solution of sub-problems, so when next time same subproblem arises, one can simply look up the previously computed solution.

Why should you apply dynamic programming to solve Markov Decision Process?

1. **Bellman equations are recursive in nature.** You can break down Bellman equation to two parts – (i) optimal behaviour at next step (ii) optimal behaviour after one step
2. You can **cache the state value and action value functions** for further decisions or calculations.

![title](img/model.JPG)

**Additional Reading**
You can read more on Dynamic Programming [here](https://web.stanford.edu/class/cs97si/04-dynamic-programming.pdf).

### Policy Iteration
In the previous session, we saw the Bellman equations of expectation and optimality. Now, we will look at how to learn the optimal policy and the corresponding state and action-value functions. There are broadly two techniques to do so:
* Policy Iteration
* Value Iteration

In the upcoming segments, you’ll learn both these methods in detail. Let's start with **Policy Iteration**.

![title](img/dp.JPG)

Policy iteration has three main steps:
1. **Initialize a policy** randomly
2. **Policy Evaluation**:

![title](img/policy_evaluation.JPG)

3. **Policy Improvement:**
* Make changes in the policy you evaluated in the policy evaluation step.
* Also known as the **control problem**.
 
The new policy you get after the policy improvement step then goes through the policy evaluation step again.

### Policy Iteration - Algorithm
Let’s now understand the **policy iteration algorithm**, i.e. policy evaluation and policy improvement, in detail.

![title](img/policy_iteration.JPG)

![title](img/policy_iteration1.png)

![title](img/policy_iteration2.JPG)

![title](img/policy_iteration3.png)

Thus, apart from initializing the policy, you also **initialize the state-value function** randomly.

Let's say that you have N states. In the policy evaluation step, you can write the value function for each state as:

![title](img/policy_iteration4.JPG)

### Policy Iteration - Comprehension
Consider a self-driving car being trained in a small **4x4** grid. All the obstacles, i.e., the pedestrian and the tree, are stationary. When the car runs into the wall (boundaries of the GridWorld), it ends up at the same location. The objective of the agent is to find an **optimal path to the goal** using policy iteration. An episode terminates if the agent runs into an obstacle or reaches the goal. The values of terminal states are 0, i.e., v(s) = 0.

**Assumption:** The environment is deterministic, i.e., given the state and the action, the agent will transition only to one particular state.

### Formulating the MDP
The most important step while solving an RL problem is formulating the MDP. You can think of MDP as the simulation of the environment, which decides the consequence of the agent's action. Let's start with it.

![title](img/mdp.JPG)

For creating the MDP environment, you need to specify states, actions, reward structure, transition probabilities and terminal states (to decide when the episode will end).

Please note that the state here is represented as (row no., column no.)

![title](img/mdp1.JPG)

Note that these transition probabilities basically represent the model of the environment.

Now, the GridWorld environment is created. It's time for the agent to learn the best **policy** to reach the Goal via policy iteration.

### Policy Evaluation - Prediction
Recall, that you obtain a system of equations in the policy evaluation step can be solved in two ways - closed form solution and iterative solution. Let's explain them in the following segment.

**Please note that** a closed form solution is one where you can solve the system of linear equations mathematically, in this case, via matrix manipulation.

![title](img/policy_evaluation1.JPG)

These equations could be solved using matrix inverse. This approach is computationally expensive and is not suitable when the number of states is huge.

![title](img/policy_evaluation2.JPG)

You will do policy evaluation for a GridWorld problem later in the session.

**Convergence in Policy Evaluation - Optional**
The state-values start to stabilise after some time. To understand why the state-values converge.

### Policy Improvement - Control
After evaluating the policy, let's now see how **policy improvement** is done.

![title](img/policy_improvement.JPG)

### Example - GridWorld
Let's try to understand the policy improvement procedure through the following example.

Consider this small 3x3 grid. Say for some arbitrary policy π; the state-value functions are as follows (the numbers in the grid represent the state-values of the corresponding state):

![title](img/policy_improvement2.JPG)

So, the best action, in this case, is to move RIGHT. The improved policy for s=(0,0) will be to move RIGHT, instead of going DOWN as according to the current policy. Similarly, you do the policy improvement for all the states.

On the next page, you will perform both the steps of policy iteration on a GridWorld problem. This will serve as a very good example to help you grasp the concept thoroughly.

### Policy Iteration - GridWorld
Let's learn to find the optimal policy using the policy iteration algorithm using the **GridWorld** setting. The agent is again the self-driving car. Please note that the state is represented as (row no., column no.).

The GridWorld is the same.

![title](img/mdp.JPG)

![title](img/policy_iteration_grid.JPG)

![title](img/policy_iteration_grid1.JPG)

![title](img/policy_iteration_grid2.JPG)

### Policy Improvement
Let's now do policy improvement using the current value-estimates.

![title](img/policy_iteration_grid3.JPG)

![title](img/policy_iteration_grid4.JPG)

Let's understand the results of policy iteration. What did the agent learn?

The optimal policy, in this case, is the ideal direction to move when at some grid position. Say, after training, you put the car at (0,2). It should now know the optimal path to reach the goal since it has learnt the value of each state.

The optimal path would be to move in a **direction where the state-value is better than the current state-value**: (0,2) ->(1,2) ->(2,2) ->(2,3) ->(3,3). So, the car should know what ("optimal") action it should take at every state.

This example illustrates how the policy iteration procedure works. You saw how you can improve a deterministic policy to arrive at an optimal solution.

### Policy Iteration for Stochastic Policy - Optional
You can start with a stochastic policy as well and follow similar steps to arrive at the optimal policy. We will not cover the stochastic policies in this course. However, you can refer here to see how to solve a similar GridWorld environment using a stochastic policy. This part is completely optional. 

### Value Iteration
Before you move ahead with **Value Iteration** algorithm, let's build an intuition of it by solving the following questions:

Consider the state (2, 0). As shown in the figure, under the current policy the agent would move UP from (2, 0), the value of (2, 0) is -12, and under the improved policy, the agent moves DOWN to (3, 0).

![title](img/value_iteration.JPG)

So you saw that you can compute the maximum value of a state by comparing the different actions possible in that state - the action which results in the max state-value is the optimal action. This is the major idea behind **Value Iteration** algorithm. 

![title](img/value_iteration1.png)

In value iteration, instead of running the policy evaluation step till the state-values converge, you perform **exactly one update**, calculating the state-values. These state-values are corresponding to the action that yields the **maximum state-action value**. The state-value function is a greedy choice among all the current actions:

![title](img/value_iteration2.JPG)

Policy improvement is an implicit step in state-value function update. The update step combines the **policy improvement** and (truncated) **policy evaluation steps**. 

### Generalised Policy Iteration (GPI)
Policy Iteration and Value Iteration are two sides of a coin.

In **policy iteration**, you complete the entire policy evaluation to get the estimates of value functions for a given a policy and then use those estimates to improve the policy. And then repeat these two steps, until you arrive at optimal state value function.

In **value iteration**, for every state update, you’re doing a policy improvement, i.e., updating the state value by picking the most greedy action using current estimates of state-values.

**Generalised Policy Iteration (GPI)** is a class of algorithms that pans out the entire range of strategies that fall between Policy Iteration & Value Iteration. A GPI is basically the following two steps running in a loop
1. Updates based on current values
2. Policy improvement

The following example will give you a gist of one such possible GPI Method. Let's take the GridWorld example. 

![title](img/gpi.JPG)

![title](img/gpi1.JPG)

After doing policy improvement for each state, the improved policy you arrive at is:

![title](img/gpi2.JPG)

Compare this with the policy we derived in Policy Iteration after the entire sweep over states. Here, we've done two sweeps and arrows are already pointing in the optimal direction. 

This is Generalised Policy Iteration. You can try any other method that range between Value & Policy Iteration and that is guaranteed to converge to optimality.

### Ad Placement Optimization (Demo) -I
Placement of ads on a website is the primary problem for companies that operate on ad revenue. The position where the ad is placed plays a pivotal role in whether or not the ad will be clicked.

Here we have the following choices:
1, Place them randomly, or
2. Place the ad on the same position

The problem with placing the ad on the same position is that the user, after a certain time, will start ignoring the space since he's used to seeing the ad at that place and will end up ignoring that particular position hereafter. This will reduce the number of clicks on ads.

The problem with the former option, i.e. placing them randomly, is that it wouldn't take optimal positions into consideration. For instance, if you place an ad beside a text, the chances of it being seen is higher than if an ad is placed randomly. It is infeasible to go through every website and repeat the procedure.

Let's see how you can solve this problem using reinforcement learning.

You can download the notebook from below:

[Ad Placement Optimization](data/Ad+Placement+Optimization_RL.ipynb)

### Why Use Reinforcement Learning
The problem of maximising click-through rate (CTR) is not a reinforcement learning problem intrinsically. We could have applied some machine learning technique but there are some bottlenecks. it requires: 
1. Huge data
2. Features
3. Tuning of many hyperparameters

You neither have a huge data, nor features. The only data you have is the position of the ad and whether or not it was clicked.

Please download the dataset from here: [Ad placement dataset from Kaggle](https://www.kaggle.com/akram24/ads-ctr-optimisation)

Let's understand the dataset.

If you were to not use reinforcement learning, you would have placed the ads randomly or at the position where maximum ads were clicked.

Let's see the results of these two policies first.

### Ad Placement Optimization (Demo) -II
Before starting with the policy iteration algorithm, let's understand the notion of an episode in this Ad placement problem. It's recommended that you analyse the code first and then watch the video.

So, the MDP for Ad placement is defined:
* State: URL ID
* Action- to place the ad at one of the 10 positions on the webpage
* Reward- If the ad is clicked on the webpage then a reward of +1, else 0
* Transition probabilities- From any webpage, he can land to any of the (n-1) web pages, where n is the total number of webpages. So, the p(s'|s,a) =1/9999

Now, let's apply Policy Iteration algorithm to the problem.

The policy iteration algorithm learns for each webpage what is the best position. For example, for webpage 1, it learns that placing the ad at Position 1, 5, 9 will result in a click. Similarly, it learns the exact solution for each of the other web pages.

This brings us to the end of the policy iteration algorithm. You can try modifying the policy iteration code to make it value iteration.

In the next session, you'll learn about the model-free methods.

## Model-Free Methods

### Introduction
Welcome to the session on Model-Free Methods.

In the last session, you learnt to solve reinforcement learning problems using Dynamic Programming (DP) through:
* Policy Iteration 
* Value Iteration

![title](img/model-free.JPG)

Although in practice, model-based methods have limited utility, the ideas of **prediction** and **control** will be reused in model-free methods as well.

In this session, we'll look into some of the model-free methods, in particular, **Monte-Carlo** & **Temporal Difference**.

### Intuition behind Monte-Carlo Methods
Let's start with Monte-Carlo methods.

Monte-Carlo method is based on the concept of **the law of large numbers**. In the upcoming segment, we will briefly explain the law.

The **law of large numbers** says that if you take a very large sample, it will give similar results as to what you would get if you would have known the actual distribution of the samples.

The **expected value** of a random variable is the weighted average over the probabilistic distribution values. It can also be thought of as the average (or mean) of a (~infinitely) large enough sample drawn from the same distribution.

![title](img/expected_value.JPG)

In the following segment, we will explain how Monte-Carlo methods exploit the above two ideas to estimate Q-values.

![title](img/expected_value1.JPG)

Let's first rearrange the Q-value function a bit.

![title](img/expected_value2.JPG)

### Monte-Carlo Prediction
**Prediction** and **control** are two integral steps to solve any Reinforcement learning problem.
* **Prediction** - evaluating the value function/policy
* **Control** - improving the policy basis the state-value function estimates

Let's start with the prediction in the Monte-Carlo method.

![title](img/monte_carlo.JPG)

![title](img/exploring_starts.JPG)

### Monte-Carlo Prediction - Demo

![title](img/monte_carlo_demo.JPG)

Similarly, you can write this for all other state-action pairs that have been visited in the episode.

Episode 2: (S3, A2, 7), (S1, A1, 5), (S7, A9, 6),(S4, A5, 4), (S6, A10,3)

![title](img/q1.JPG)

![title](img/ans1.JPG)

![title](img/q2.JPG)

![title](img/ans2.JPG)

Let's try running for one more episode:

![title](img/monte_carlo_demo1.JPG)

### Monte-Carlo Control
In the last segment, you learnt how to **evaluate Q-function for a given policy**. Now, you can use these q-values for the **control** problem. 

Recall that the control problem is to improve the existing policy in order to find an optimal policy. Let's see how you can do it.

**Please note** that A(s) represents the action space, i.e. the total number of actions that are possible in any given state. We talks about the ϵ-greedy approach in the following segment, you will study that later on this page.

![title](img/monte_carlo_control.JPG)

![title](img/monte_carlo_control1.JPG)

**Please note** that in the upcoming segment:
* **Exploration** refers to the scenario where you do not follow the actions suggested by the policy. Instead, you go and explore some other actions.
* **Exploitation** refers to the scenario where you religiously follow the actions suggested by the policy.

Let's understand what it means to have a **ϵ-greedy policy**.

![title](img/monte_carlo_control2.JPG)

![title](img/monte_carlo_control3.JPG)

### Off Policy
In the previous few segments, you learnt that the **control problem** seeks the **best action values** for the agent to behave optimally.

However, the agent has to behave non-optimally in order to explore other actions (i.e. the actions that are not given by the policy), to find better action than already existing one.

How does the agent learn the optimal policy while it is continuing to explore? In the following segment we will explain this in detail.

A way to handle the dilemma of exploring and exploitation is by using two policies:
* **Target Policy**: the policy that is learnt by the agent and that becomes the optimal policy
* **Behaviour Policy**:  the policy that is used to generate the episodes and is more exploratory in nature

This is called **off-policy** because the agent is learning from data 'off' the behavior policy.

In another case, if the (optimal) policy learnt and the policy that is used to generate episodes are the same, it is called **on-policy** learning. 

![title](img/off-policy.JPG)

![title](img/importance_sampling.JPG)

### Temporal Difference
In the earlier segments, you learnt about **Monte-Carlo Prediction & Control**. There are some inherent issues with Monte-Carlo. To start with, you need to wait until the end of the episode to update the expected rewards for a state-action pair.

In the following segment, we explain some other practical problems with Monte-Carlo and how these issues led to the development of **Temporal Difference** learning.

Similar to Monte-Carlo, **Temporal Difference (TD)** methods also learn directly from experience without an explicit model of the environment.

The major difference between these two methods is:
* In Monte Carlo methods, you need to wait until the end of the episode and then update the q-value
* In TD methods, you can update the value after every few time steps. This update mechanism is often advantageous since it gives the agent early signals if some of the eventual states are disastrous, and so the agent avoids that path altogether.

Let's understand this through an example.

Say there are two agents in a **chemical plant**. One is learning using Monte-Carlo while the other is learning using TD. Each day is a new episode for them. One day, they both **changed the similar valves** (action). The pressure in one of the tube increased too much and the yield of the product started decreasing (the reward started becoming negative). The TD agent will update the q-value of this action (i.e. reduce it immediately) and will instead take some other action that will increase the future rewards. On the other hand, the Monte-Carlo agent will continue taking actions according to its policy. The pressure could increase beyond a threshold level leading to a blast. After this episode, the MC agent would realise that the action (of turning that valve) was quite bad and update the policy accordingly, though the TD agent cleverly avoided all the trouble altogether.

In the next segment, you'll learn how to update the q-value at each step in TD.

### Q-Learning
Let's now learn to write the **Temporal Difference** (TD) update equation.

![title](img/temporal_difference.JPG)

Let's understand the TD algorithm using a small example:

Say you played the game of Chess **10,000 times** (episodes) over a period of 5 years. You have learnt that when you take a particular action from some board position, you always end up getting a high reward - the q(s, a) is 100 for that state-action pair. Today, when you played the game, you got a total reward of 70 (starting from the same state-action pair). How will you update the value of q(s, a)? You cannot directly make it 70 as you can't ignore the learning of 5 years. Also, intuitively, you know that the updated value should be lesser than 100 (since the latest reward of 70 is lesser than 100). So you'll update the q-value incrementally in the direction of the new, latest learning.

![title](img/temporal_difference1.JPG)

TD update equation relies on a similar idea. There are many different versions of TD update. The only difference in all updates is the calculation of current value. In this module, we'll cover only **Q-learning**. 

### Q-Learning

![title](img/q-learning.JPG)

In the next lecture, we will explain how you can write the update equations for different state-action pairs. 

Q-learning learns the optimal policy 'relentlessly' because the estimates of q-values are updated based on the q-value of the next state-action pair **assuming the greediest action will be taken subsequently**.

If there is a risk of a large negative reward close to the optimal path, Q-learning will tend to trigger that reward while exploring. And in practice, if the mistakes are costly, you don't want Q-learning to explore more of these negative rewards. You will want something more conservative.

There are other approaches like  SARSA, Double Q-learning, which are more conservative and avoid high risk. Please note that these two techniques are provided as optional reading for interested students.

**Additional Reading**
* This segment covers only TD(0) update, which is only a one-step update, i.e., the agent takes one step and update q-value. In general, you could consider λ steps. And such methods are called TD(λ). You can read more about it [here](https://amreis.github.io/ml/reinf-learn/2017/11/02/reinforcement-learning-eligibility-traces.html).
* Apart from the update step difference, there are some other implicit differences between Monte-Carlo and Temporal Difference. You can read more [here](https://stats.stackexchange.com/questions/336974/when-are-monte-carlo-methods-preferred-over-temporal-difference-ones).

### Q-Learning Pseudocode
In this segment, you'll learn the pseudocode of Q-Learning. The update equation for Q-learning is: 

![title](img/q-learning1.JPG)

![title](img/q-learning2.JPG)

![title](img/q-learning3.JPG)

Let's look at the value iteration calculations in more detail in the next segment.

### Cliff Walking Demo
In this segment, you'll learn to solve for an optimal stochastic policy using Monte-Carlo and Q-Learning algorithm for GridWorld environment. First, let's see how Monte-Carlo (first-visit) agent works.

![title](img/cliff_walking.png)

![title](img/cliff_walking1.JPG)

![title](img/cliff_walking2.JPG)

![title](img/cliff_walking3.JPG)

![title](img/cliff_walking4.JPG)

So, this is how Q-learning works. This demonstration was just to get you more familiar with the calculations of Monte-Carlo and Q-Learning. It's easier to write code once you get the basic math for solving these algorithms.

### Ad Placement Optimization Demo -Q Learning

![title](img/ad_placement.JPG)

[AD Placement - Q Learning](data/Ad+Placement+Optimization_RL.ipynb)

Similar to Dynamic Programming, Q-Learning agent has learnt the exact solution to the problem. The policy at each webpage is placing the ad at a position where the user has clicked.

An important thing to notice is that ϵ is decaying with the number of episodes. It is important for the agent to both explore and exploit. This decay rate allows the agent to make completely random moves to explore the state space maximally, and then to settle down to a fixed exploration rate. 

### OpenAI Gym -Taxi v2
Founded by Elon Musk and Sam Altman, **OpenAI** is a non-profit research company that is focussed on building out AI algorithms. **OpenAI Gym** is a toolkit for developing and comparing reinforcement learning algorithms. Gym provides different environments to implement various reinforcement learning algorithms.
 
This segment is to make you familiar with OpenAI Gym. You'll learn to use these environments and try running RL algorithms on it.  In the following segment, we will walk you through OpenAI Gym environment -Taxi-v2 and how to use it. You can download the Jupyter notebook for your reference.

[Open AI Gym Env](data/Taxi-v2.ipynb)

Now, we has explained the MDP for Taxi-v2. It's an exercise for you to apply Q-Learning on it. We've shared the Q-Learning and SARSA implementation code with you. But, it's important that you try writing the code on your own. It will help later when you go to the next module on **Deep Reinforcement Learning**.