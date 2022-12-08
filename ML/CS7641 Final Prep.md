# Lesson outline
- [Module: UL2 - Clustering](#1)
- [Module: UL3 - Feature Selection](#2)
- [Module: UL4 - Feature Transformation](#3)
----------------------------------------------
- [Module: RL1 - Markov Decision Processes](#4)
- [Module: RL2 - Reinforcement Learning](#5)
- [Module: RL3&4 - Game Theory](#6)

<h1 id="1">Module: UL2 - Clustering</h1>

### The Clustering Problem ###
Input:  
1) A set of objects X   <br/>
2) A distance metric D(point_a, point_b) defining inter-object distances such that D(x,y) = D(y,x) where x,y ∈ X <br/>

Output: A partition of the objects such that P_D(x) = P_D(y) if x and y belong to the same cluster<br/>

The distance metric defines the similarity, like KNN. But it doesn't measure/differentiate good/bad clustering. The clustering is algorithm-driven. Each clustering probelm is its own probelm.<br/>

### Single Linkage Clustering (SLC) ### 
SLC is hierarchical agglomerative clustering (HAC) of algorithm. Steps: <br/>
• Consider each object a cluster (n objects).<br/>
• Define inter-cluster distance as the distance between the closest points in the two clusters.<br/>
• Merge the two closest clusters.<br/>
• Repeat 𝑛 − 𝑘 times to make 𝑘 clusters.<br/>

**Running time of SLC**: simplest case of SLC 𝑂(𝑛^3): evaluate and compare n^2 pairs of points at least k times (k = n in the worst case)<br/>
**Properties**: It is deterministic, and it actually equates to a minimum spanning tree algorithm if we treat distances as edge lengths<br/>
**Issues with SLC**: Clusters lean towards connected neighbouring points, which is not necassarily accurate. It might end up with wrong clusters due to noise<br/>
How is median linkage clustering being different from average/mean linkage clustering? The median is good when the number on the distances don't matter, just their ordering. median is a non-metric statistic. Average is a metric statistic which details of number matters<br/>

Hierarchical clustering. Distance metrics: https://towardsdatascience.com/understanding-the-concept-of-hierarchical-clustering-technique-c6e8243758ec


### K Means Clustering ### 
K-means clustering: https://neptune.ai/blog/k-means-clustering<br/>
K-means clustering wiki:https://en.wikipedia.org/wiki/K-means_clustering<br/>
StatQuest: K-means clustering: https://www.youtube.com/watch?v=4b5d3muPQmA<br/>

K-means is a centroid-based clustering algorithm, where we calculate the distance between each data point and a centroid to assign it to a cluster. The goal is to identify the K number of groups in the dataset. Algorithm steps: <br/>
• Pick K centres at random<br/>
• Each centre claims it's closest points<br/>
• recompute the centres by averaging the clustered points<br/>
• repeat until convergence<br/>

**Properties**: <br/>
1. k-means always converge: The standard algorithm (naive k-means) with the least squared Euclidean distance will converge to the local minimum. However, using a different distance function other than (squared) Euclidean distance may prevent the algorithm from converging. Various modifications of k-means such as spherical k-means and k-medoids have been proposed to allow using other distance measures<br/>

𝑘-means as an optimization problem: maximize score, minimize error <br/>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul2.JPG?raw=true">
</p>

𝑘-means in Euclidean space:<br/>
<p align="center" width="100%">
    <img width="90%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul1.JPG?raw=true">
</p>

Could we be monotonically non-increasing in error forever? We could, but not in this KMean case. monotonically non-increasing function is a function that never goes bigger. so you could end up in a case where you hit some point like zero error and keep going. Why it wouldn't happen here? Because there are finite number of configurations as there are finite number of objects and finite number of labels they can have. Once you've chosen a label for a bunch of objects, the centers are defined deterministically from that. Even we have inf space as we loop back and forth, if we don't move the partition, then the centers are going to be where they were. So the centers are quite constrainted even though it is continuous. The tricky part is when the tie happens.There has to be some ways to break the tie to get the same answer. <br/>
Breaking tie consistently (give a rule) is going to guarantee that we are at least don't spin around without improving.<br/>
In summary, if we have a finite number of configurations, if we always break ties consistently and we never go into a configuration with a higher error, then I will never repeat configurations and never go back to a configuration that we were at before. So at some point, I will run out of configurations because there are finite number of them. so it converges in a finite time<br/>

2. Each iteration is polynomial 𝑂(𝑘𝑛)  (K centers and n points to reassign them) <br/>
3. Finite (exponential) iterations 𝑂(𝑘^𝑛) <br/>
4. Error decreases (if ties are broken consistently) <br/>
5. It can get stuck in the local minima if the initial choices are bad or don't cover the space well. This is similar to converging to a local optima. One solution to this problem is to use random restarts <br/>

**Issues with K Means**: 
1. sensitive to initialization and outliers and get stuck in the local minima. Solutions: 1)Random restarts 2)Kmean++ <br/>
2. it only finds “spherical” clusters. In other words, because we rely on the SSD as our “error function,” the resulting clusters try to balance the centers to make points in any direction roughly evenly distributed <br/>

K Means VS Hierarchical clustering: K-means clustering specifically tries to put the data into the number of clusters you tell it to. Hierarchical clustering just tells, pairwise, what two things are most similiar. <br/>


### Soft Clustering ### 
Soft clustering attaches cluster probability to each point, instead of a specific cluster<br/>
Assume the data was generated by:<br/>
- Select one of 𝑘 possible Gaussians (Fixed know variance) uniformly.<br/>
- Sample 𝑥𝑖 from that Gaussian.<br/>
- Repeat 𝑛 times<br/>

So we are assume n points were selected from k uniformally-selected Gaussian distributions.<br/>

Goal: Find a hypothesis ℎ = 〈𝜇1, … , 𝜇𝑘〉 (𝜇1, … , 𝜇𝑘 are Gaussian means) that maximizes the probability of the data (Maximum likelihood)<br/>

Maximum Likelihood Gaussian: The Maximum Likelihood mean of the Gaussian 𝜇 is the mean of the data.<br/>
Single ML Gaussian: Suppose k = 1 for the simplest possible case. What Gaussian maximizes the likelihood of some collection of points? it is simply the Gaussian with µ set to the mean of the points!<br/>
Extending to Many: With k possible sources for each point, we introduce hidden variables for each point that represent which cluster they came from. They are hidden because obviously we don’t know them: if we knew that information we wouldn’t be trying to cluster them. Now each point x is actually coupled with the probabilities of coming from the clusters. This leads us to Expectation Maximization.<br/>

 ### Expectation Maximization ### 
EM assigns a cluster probability to each point and use the new 𝜇𝑗 to re-compute 𝐸.<br/>
• Expectation: 𝐸 defines the probability that element 𝑖 was produced by cluster 𝑗 <br/>
• Maximization: 𝜇𝑗 defines the mean of cluster 𝑗 <br/>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul3.JPG?raw=true">
</p>

Kmean improves the error metric, and EM improves the probabilistic metric.<br/>
EM is not forced to make a decision about the overlapping samples, as soft clustering. But one of the consequence is that even one sample very clearly belong to one cluster, they all have some non-zero probability that it belong to the other cluster. Because Gaussian have infinite extent even the points are far away from the center. This tell us the chance is low but non-zero.<br/>


**Properties of EM**<br/>
• **Monotonically non-decreasing likelihood**: Each time the iteration of EM runs, the likelihood of the data is monotonically non-decreasing. It is not getting worse. Generally it is finding higher and higher likelihoods and moving in a good direction. 
<br/>
• **Does not converge (practically converge)**: However, monotonically non-decreasing likelihood doesn't mean the algorithm has to converge. Although in practice, it always converge<br/>
• **Will not diverge**: Even it doesn't gaurantee converge, it cannot diverge. The number cannot blow up and become infinitely large, because it is working in the space of probablities. This is a difference in Kmeans which has finite number of configurations and k means Kmean never gets worse in error metric, as long as you have some ways to break ties, eventually you have to stop. That's how you get convergence. In EM, the configurations are probabilities, which is infinite number. You never do worse, but you are trying to move closer and closer. You could keep moving closer every single time, but because of the infinite number of configurations,the step by which you get better could keep getting smaller. So you never actually approach the final best configuration. <br/>
• **Can get stuck**: In practice it is very common. The local optimal problem. solution: random start <br/>
• **Works with any distribution (if 𝐸 and 𝜇 are solvable)**: Nothing specific with Guassian. Different distributions can be used to solve E and 𝜇. Usually it is the case that estimation step is expensive and difficult because it invovles probabilistic inference, like Bayes net. And Maximization step is just counting things. In general, it is harder to do E than M. EM is a small matter of mathematical algorithm derivation <br/>


Check EM part from http://stanford.edu/~cpiech/cs221/handouts/kmeans.html<br/>
K-Means is really just the EM (Expectation Maximization) algorithm applied to a particular naive bayes model.<br/>

External: Clustering (4): Gaussian Mixture Models and EM https://www.youtube.com/watch?v=qMTuMa86NzU <br/>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul3_1.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul3_2.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul3_3.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul3_4.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul3_5.JPG?raw=true">
</p>
k-means is a special case of expectation maximization: variances are all equal, and there is no covariance <br/>
Consider the two clusters in the right hand side of the image. There are two clusters both centred at the same mean. K Means has diffuciulty with this. GMMs are an extension of the Kmeans models, where clusters are modelled using gaussian distributions. Where each cluster will have not only a means but also a covariance which helps explain their ellipsoidal shape. We can then fit the model by maximizing the likelihood of the observed data. We do this with an algorithm called EM, for expecation maximization, which assigns each sample to a cluster with a soft probability.<br/>


### Clustering Properties ###
**Desirable Properties for any given clustering approach**:
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul4.JPG?raw=true">
</p>
1. Richness: There are some distance metrics that would cause your clustering algorithm to produce that clusters. So all inputs are valid and all outputs are valid. The algorithm should produce wahtever is appropraite and not limit to what it can express. The alternative of richness (non-richness): there are certain clusters you just cannot produce.<br/>
2. Scale-invariance: Doubling the distance or changing the unit shouldn't change what the clustering is. it should be invariant to what the space of the point is, assuming that we keep the relative distances the same<br/>
3. Consistency: Shrinking intra-cluster distances (Moving points towards each other, similiar become more similar) and expanding inter-cluster (Moving clusters away from each other, not similar become less similar) distances doesn’t change the clustering. Given a particular clustering, we would imagine that by “compressing” or “expanding” the points within a cluster (think squeezing or stretching a circle), none of the points would magically assign themselves to another cluster. In other words, shrinking or expanding intracluster distances should not change the clustering.<br/>

<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul5.JPG?raw=true">
</p>
Ans: In this SLC stop when case 1) fixed number of cluster doesn't have richness because richness would allow for one cluster or it could have one cluster or it could have n clusters or it could have n/3 clusters or etc. but here we are forced it to have n/2 clusters, so it cannot represent all possible clusters. 2) we can group whatever combination we want, but if we multiply everything by theta, then I have n. But if I have n in the beginning then divide theta, I would have one. It is not scale-invariance 3) W is max function to normalize the distance. This is still scale-invariance as larger unit also makes larger W, which means the scale is undone. However, if expanding inter-cluster, W will be further and change the cluster. eg, theta divide by inf will make no point be able to cluster with the other. This is not consistent.<br/>


**Impossibility Theorem**<br/>
Impossibility Theorem: There’s no clustering algorithm that can achieve these three properties.<br/>
Jon Kleinberg: https://www.cs.cornell.edu/home/kleinber/nips15.pdf<br/>
https://jeremy9959.net/Blog/KleinbergsClusteringTheorem/<br/>
k-means and EM has the properties of scale - invariance and consistency but not richness because k determines # of clusters. Essentially, if were to introduce a new distance function and arrive to the same cluster configuration (partition), then you would satisfy the consistency property. From Kleinberg paper, Section 4: "We show here that for a fairly general class of centroid-based clustering functions, including k-means and k-median, none of the functions in the class satisfies the Consistency property. This suggests an interesting tension between between Consistency and the centroid-based approach to clustering, and forms a contrast with the results for single-linkage and sum-of-pairs in previous sections. " <br/> 

**Summary**<br/>
Soft cluster (EM) vs Hard cluster (SLC, Kmean)<br/> 
Terminates in polynomial time (SLC) vs Terminates not in polynomial time (Kmean, EM)<br/> 


<h1 id="2">Module: UL3 - Feature Selection</h1>
**The Feature Selection Problem**:
Goals: 1) Knowledge Discovery, Interoperability & Insight. 2) Avoid Curse of Dimensionality, The amount of data you need grows exponentially with the number of features you have. By reducing our dimensions we can reduce the difficulty of the problem. In summary, with proper feature selection, not only will you understand your data better, but you will also have an easier learning problem<br/> 

Suppose we have N features and we want to reduce to M features where M <= N. How hard is this problem?<br/> 
Is it Polynomial (includes linear and quadratic) or exponential? ans: exponential<br/> 
To do this we need to come up with some sort of function that returns a score. We could choose M from N. It turns out this is a well known problem which NP-Hard. It's exactly because you have to find all possible subsets<br/> 


**Approaches to Feature Selection**:
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul5.JPG?raw=true">
</p>
**Filtering**: Given a set of features, apply a search algorithm to produce fewer features to be passed to the learning algorithm. In summary, filtering directly reduces to a smaller feature set and feeds it to a learning algorithm<br/> 
Pros: 1) Faster than wrapping <br/> 
Cons: <br/> 
1) Can also be slower because you look at features in isolation <br/> 
2) Doesn't account for the relationships between features<br/>   
3) Ignores the learning process (no feedback)<br/> 
We can use an algorithm that can select the best features, like Decision Tree, then use another algorithm, with another inductive bias, for learning. <br/>
We can use different criterion to evaluate the usefulness of a subset of features. This is where the domain knowledge comes in: 1. Information gain 2. Variance 3. Entropy 4. Eliminate dependent features<br/> 
What about using a Neural Network in search box and pruning the lowest weighted features? You could prune correlated or dependent features.<br/> 

Note that you can use the labels in the search algorithm. It's not considered cheating to understand what the labels are in the search box  before passing into the learner.<br/> 
In fact Information Gain is often used in the search box. You could even go so far as using a decision tree (algorithm used in search box) to determine what should be passed to a Neural Network (actual learner, learner box). Basically, you are using the inductive bias of the decision tree to choose features, but then you use the inductive biasof your other learner in order to do learning. If the learner box is KNN, which suffers from the curse of dimensionality because it doesn't know which features are important. then this search box decision tree is good at figuring out which features are importent, and gives soem hint to KNN.<br/> 


**Wrapping**: Given a set of features, apply a search algorithm to produce fewer features, pass them to the learning algorithm, then use the output to update the selected set of features. Here the ML quality is passed back as a quasi-score value which is then used by the algo to determine the final features. In summary, wrapping interacts with the learning algorithm directly to iteratively adjust the feature set <br/>
Pros: Takes into account model bias, score and learning  <br/>
Cons: Very slow <br/>
We can use different techniques to search: Local searching can be useful. We can use approaches directly from randomized optimization algorithms if we treat the learner's result as a fitness function. Other viable options include forward search and backward search. Note that we should avoid an exhaustive search here as this can make the problem under a worst case scenario<br/>
1. Randomized Optimization algorithms<br/>
2. Forward Selection: Select a feature and evaluate the effect of creating different combinations of this feature with other features. Stop once you stagnate. This is similar to Hill Climbing.<br/>
3. Backward Elimination: Start with all the features and evaluate the effect of eliminating each feature<br/>


**Describing Features**:
When it comes to determining which features we should use, it’s necessary to differentiate them based on their viability in a general, statistical sense as well as in a learner-specificsense. We usually care more about usefulness, but relevance is what we generally use to get there.<br/>
Relevance  ∼  Information <br/>
Usefulness  ∼  Error given a Model/learner<br/>

**Relevance**<br/>
Let B.O.C. = Bayes Optimal Classifier<br/>

• xi  is strongly relevant if removing it degrades B.O.C.<br/>
• xi  is weakly relevant if<br/>
    1) Not strognly relavant<br/>
    2) ∃ a subset of features S, such that adding  xi  to S improves B.O.C.<br/>
• xi  is otherwise irrelevant<br/>

**Usefulness**<br/>
• Relevance measures effect on B.O.C.<br/>
• Usefulness measures effect on a particular predictor<br/>



<h1 id="3">Module: UL4 - Feature Transformation</h1>

<h1 id="4">Module: RL1 - Markov Decision Processes</h1>

<h1 id="5">Module: RL2 - Reinforcement Learning</h1>

## Extra, Great content! Intro to RL, from CIS 522 [Deep Learning @ Penn](https://www.youtube.com/watch?v=cVTud58UfpQ&list=PLYgyoWurxA_8ePNUuTLDtMvzyf-YW7im2&index=2)
### CIS522.1 Intro to RL

RL: given observations and occasional rewards as the agent performs sequential actions in an environment. Compared to supervised learning and unsupervised learning, you no longer have dataset given in advance. Instead, you receive your data as the agent performs some sequential actions in an environment. That process of performing sequential actions generates some obversations accompained by rewards. So there is no label but the reward essentially tells you whether the actions that you performed were good or not.<br/><br/>

The aim of RL is to make sequential decisions in an environment. For example, driving a car. How to learn to do these things? 1) RL assumes only occasional feedback, for example, a car crash 2) RL aims to use this feedback to learn through trail and error, as cleverly as possible.<br/><br/>

Main Idea/Turn-by-turn abstraction: <br/>
Agent + environment. Agent's goal is to maximize expected rewards. <br/>
Step1) Agent receives observations(state of the environment s_t) and feedback(reward r_t) from the world. Often, agent observers features, rather than the true state. For example, the car may not observe a pedestrain is hidden behind a car. In addition, we don't always get a reward at time t as reward is only occasional. <br/>
Step2) Agent emits an action a_t into the environment.  <br/>
Step3) Loop back to Step1, gets updated state and reward<br/>

With this abstraction, the goal of RL is to learn a policy π(s): S -> A (mapping from states to actions) for acting in the environment<br/><br/>

Characteristics of RL probelms: <br/>
1) No supervision, only (occasional) rewards as feedback <br/>
2) sequential decision making. Data is generated as sequences, not i.i.d <br/>
3) Training data is generated by the learner's own behavior<br/><br/>

Key Probelms specific to RL: <br/>
1) Credit assignment: which decisions were the good/bad ones <br/>
2) Exploration VS Exploitation: Yes, trail-and-error, but how to pick what tor try?<br/><br/>

When do we no need to worry about sequential decision making:your system is making a single isolated decision that does not affect future decision, e.g. classficication, regression<br/>

When should we worry about sequential decision making: <br/>
1) limited supervision: you know what you want, but not how to get it <br/>
2) actions have consequences<br/><br/>

### CIS522.2 Markov Decision Processes
Toy example - grid world - Deterministic grid world vs Stochastic grid world (Details skipped)<br/><br/>
An MDP(S,A,P,R) is defined by: <br/>
1) Set of States s ∈ S. In the grid world,it would be all the different configurations of the environment <br/>
2) Set of actions a ∈ A. In the grid world,it would be N,S,E,W<br/>
3) (State) Transition Function P(s'|s,a). It's the probability of transitioning into a new state s' given s and a. Also called the dynamics model or just the model<br/>
4) Reward function R(s,a,s') or R(s) <br/>
In RL, we typically do not know the true functions P(.) or R(.), Instead we only get samples from them. So we have to learn from trail and error <br/><br/>
The Markov Property: Given the present, the future and the past are independent<br/>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition1.JPG?raw=true">
</p>


### CIS522.3 Solving MDPs
To solve an MDP(S,A,P,R) means to find the optimal policy π*(s): S -> A. Optimal means following this policy will maximize the total reward/utility (on average)<br/>
In RL, P and R are unknown. But first let's assume we know the whole MDP<br/>
MDP Search Trees: Each MDP state has an associated expectimax-like tree of future outcomes from various actions. <br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition2.JPG?raw=true">
</p>

Define Utility<br/>
At each step, agent chooses an action to maximize expected rewards. So we must consider utility over sequences of rewards List(r_t, r_t+1, r_t+2 .... r_inf)<br/>
Probelm: Infinite sequences yield infinite rewards<br/>
Solutions:<br/>
1) Finite horizon - episode terminates after a fixed number of steps. This yields nonstationary policies that vary depending on the amount of time left. Note from online, Episodic tasks are the tasks that have a terminal state (end). In RL, **episodes** are considered agent-environment interactions from initial to final states (Episode: All states that come in between an initial-state and a terminal-state). For example, in a car racing video game, you start the game (initial state) and play the game until it is over (final state). This is called an episode. Once the game is over, you start the next episode by restarting the game, and you will begin from the initial state irrespective of the position you were in the previous game. So, **each episode is independent of the other**. In a continuous task, there is not a terminal state. Continuous tasks will never end. For example, a personal assistance robot does not have a terminal state.<br/>
2) Absorbing state - guarantee that every policy reaches a terminal state. So you can engineer your mdp such taht it has the absorbing state no matter what you do<br/>
3) Discounted rewards (most generally used) - (uncertain) future rewards are worth exponentially less than current rewards.<br/><br/>

Discounted rewards<br/>
Idea: uncertain future rewards are worth exponentially less than the current reward. So future rewards matter less to the decision than the more recent rewards<br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition3.JPG?raw=true">
</p>

MDP quantities so far<br/>
1) policy = choice of action for each state<br/>
2) utility/return = sum of discounted rewards<br/><br/>

stackexchange: Understanding the role of the discount factor in reinforcement learning https://stats.stackexchange.com/questions/221402/understanding-the-role-of-the-discount-factor-in-reinforcement-learning<br/>
The fact that the discount rate is bounded to be smaller than 1 is a mathematical trick to make an infinite sum finite. This helps proving the convergence of certain algorithms.<br/>

In practice, the discount factor could be used to model the fact that the decision maker is uncertain about if in the next decision instant the world (e.g., environment / game / process ) is going to end.<br/>

For example:<br/>
If the decision maker is a robot, the discount factor could be the probability that the robot is switched off in the next time instant (the world ends in the previous terminology). That is the reason why the robot is short sighted and does not optimize the sum reward but the discounted sum reward.<br/>


### CIS522.4 The Bellman Equation
combined with https://www.datascienceblog.net/post/reinforcement-learning/mdps_dynamic_programming/<br/>
In a simplified setting MDP(S,A,P,R,γ) where we know S,A,P,R,γ. <br/>

**State-value function**<br/>
 State Value Functions of Policies is a function of both the state and policy that you are currently following.<br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition4.JPG?raw=true">
</p>

**Policy**: Which actions the agent should execute in which state. A policy, π(s,a), determines the probability of executing action a in state s. In deterministic environments, a policy directly maps from states to actions.<br/>
**State-value function**: The expected value of each state with regard to future rewards<br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition5.JPG?raw=true">
</p>


**Action-value function**<br/>
Compared to the State-value function, we are no longer computing the expected future rewards conditions on being in a state, instead, we are computing the expected future rewards conditions both on being in a state and perform a particular action from that state. The computation result is not true state, but Q-state corresponding to kind of an imaginary state that exists after having executed a particular action a from state s. So the Action-value function is also called Q-function <br/>

If you have the optimal Q*, then you can easily determine what the policy π* is. Remember Q* is telling you what is the value/utility to be gained by executing an action a at the state s, and then it follows the optimal policy. That means, you don't have to worry about things that happen after that first step, because afterwards you are guaranteed to be following the optimal policy that's what the definition of Q* is. So now what you need to optimize over is the first action a. That would give you the policy. <br/>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition6.JPG?raw=true">
</p>
Action-value function: The expected value of performing a specific action in a specific state with regard to future rewards<br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition7.JPG?raw=true">
</p>


**Bellman Equations**<br/>
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition8.JPG?raw=true">
</p>

### CIS522.5 Value and Policy iteration (solving MDPs with Known P and R)
<!-- Combined with articles: Not done yet<br/>
1) https://medium.com/@ngao7/markov-decision-process-basics-3da5144d3348
2) https://medium.com/@ngao7/markov-decision-process-policy-iteration-42d35ee87c82
3) https://medium.com/@ngao7/markov-decision-process-value-iteration-2d161d50a6ff#8adf -->

Bellman equation gives us a recursive definition of the optimal value. We can slove iteratively via dynamic programming<br/>

**Value Iteration**: <br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition10.JPG?raw=true">
</p>

<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition9.JPG?raw=true">
</p>
(https://towardsdatascience.com/value-iteration-to-solve-openai-gyms-frozenlake-6c5e7bf0a64d)

**Policy Iteration**: <br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition11.JPG?raw=true">
</p>

Simply means to compute the value function corresponding to a policy. The value function is associated with some particular policy. The optimal value function is associated with the optimal policy. In value iteration, we were dealing throughout with the optimal value function. But now with policy iteration, we try to compute value function for some random policies. It turns out there is actually an existing version of the update rule for bellman equation. Just like we use the bellman equation for optimal value functions in value iteration, we use bellman equation for arbitrary policies in policy iteration   <br/>

<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition12.JPG?raw=true">
</p>

Go back and forth between policy evalution and policy improvement until convergence. Example see below chart. <br/>
Think about how it relates to value iteration. What happens if you do only one iteration in the policy evaluation? This means in the policy evaluation setup, you don't wait for value to converge. Just do one iteration of policy evaluation. Actually if you do so, this reduce policy iteration to the value iteration. <br/>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition13.JPG?raw=true">
</p>

**Comparison of methods for solving MDPs**: <br/>
Value iteration: Each iteration updates both utilities (explicitly, based on the current utilities) and the policy (possibly implicitly, based on the current utilities)<br/>
Policy Iteration: Several iterations to update utilities for a fixed policy, occasional iterations to update policies <br/>
Hybrid methods (asynchronous policy iteration): Any sequences of partial updates to either policies or utilities will converge if every sate is visited infinitely often<br/>


### CIS522.6 Temporal Differencing (TD) and Q Learning

**Temporal Differencing (TD)**: <br/>
Look at policy evaluation under fully known MDPs, we evaluate any policy by assigning values to each state under a fix policy. Here transition probability P and R are known.<br/>
So, how to extend this policy evaluation to when the functions P and R are unknown and only revealed gradually through experience? Everytime when you perform an action in the environment, you will get a sample from each of these distributions. In particular, the state that you end up at is observable to you. This gives you information about transition function P and you do get corresponding reward from the environment. So you get samples from the two unknown functions.<br/>
<br/>


<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition14.JPG?raw=true">
</p>


**Q-learning**: <br/>
Same as policy evaluation TD, setting alpha as 1 is a bad idea because the update is completely noisy. Instead, small alpha let us not only use the new single sample but also bring in the knowledge Q(s,a) from the past interactions.<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition15.JPG?raw=true">
</p>

<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rladdition16.JPG?raw=true">
</p>


<h1 id="6">Module: RL3&4 - Game Theory</h1>

