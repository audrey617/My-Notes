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
• Pick K center at random<br/>
• Each center claims its closest points<br/>
• recompute the center by averaging the clustered points<br/>
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

Could we be **monotonically non-increasing in error** forever? We could, but not in this KMean case. monotonically non-increasing function is a function that never goes bigger. so you could end up in a case where you hit some point like zero error and keep going. Why it wouldn't happen here? Because there are finite number of configurations as there are finite number of objects and finite number of labels they can have. Once you've chosen a label for a bunch of objects, the centers are defined deterministically from that. Even we have inf space as we loop back and forth, if we don't move the partition, then the centers are going to be where they were. So the centers are quite constrainted even though it is continuous. The tricky part is when the tie happens.There has to be some ways to break the tie to get the same answer. <br/>
Breaking tie consistently (give a rule) is going to guarantee that we are at least don't spin around without improving.<br/>
In summary, if we have a finite number of configurations, if we always break ties consistently and we never go into a configuration with a higher error, then I will never repeat configurations and never go back to a configuration that we were at before. So at some point, I will run out of configurations because there are finite number of them. so it converges in a finite time<br/>

2. Each iteration is polynomial 𝑂(𝑘𝑛)  (K centers and n points to reassign them) <br/>
3. Finite (exponential) iterations 𝑂(𝑘^𝑛) <br/>
4. Error decreases (if ties are broken consistently) <br/>
5. It can get stuck in the local minima if the initial choices are bad or don't cover the space well. This is similar to converging to a local optima. One solution to this problem is to use random restarts <br/>

**Issues with K Means**: 
1. sensitive to initialization and outliers and get stuck in the local minima. Solutions: 1)Random restarts 2)Kmean++ <br/>
2. it only finds “spherical” clusters. In other words, because we rely on the SSD as our “error function,” the resulting clusters try to balance the centers to make points in any direction roughly evenly distributed <br/>
3.  Middle point can end up with either cluster depending on how ties are broken and how the initial centers are chosen. solution: probability. leads to soft clustering<br/>


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
• **Will not diverge**: Even it doesn't gaurantee converge, it cannot diverge. The number cannot blow up and become infinitely large, because it is working in the space of probablities. This is a difference in Kmeans which has finite number of configurations and Kmeans never gets worse in error metric, as long as you have some ways to break ties, eventually you have to stop. That's how you get convergence. In EM, the configurations are probabilities, which is infinite number. You never do worse, but you are trying to move closer and closer. You could keep moving closer every single time, but because of the infinite number of configurations,the step by which you get better could keep getting smaller. So you never actually approach the final best configuration. <br/>
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
Consider the two clusters in the right hand side of the image. There are two clusters both centred at the same mean. K Means has difficulty with this. GMMs are an extension of the Kmeans models, where clusters are modelled using gaussian distributions. Where each cluster will have not only a means but also a covariance which helps explain their ellipsoidal shape. We can then fit the model by maximizing the likelihood of the observed data. We do this with an algorithm called EM, for expecation maximization, which assigns each sample to a cluster with a soft probability.<br/>


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
"k-means and EM has the properties of scale - invariance and consistency but not richness because k determines # of clusters." statement is wrong. Essentially, if were to introduce a new distance function and arrive to the same cluster configuration (partition), then you would satisfy the consistency property. From Kleinberg paper, Section 4: "We show here that for a fairly general class of centroid-based clustering functions, including k-means and k-median, none of the functions in the class satisfies the Consistency property. This suggests an interesting tension between between Consistency and the centroid-based approach to clustering, and forms a contrast with the results for single-linkage and sum-of-pairs in previous sections. " <br/> 

**Summary**<br/>
Soft cluster (EM) vs Hard cluster (SLC, Kmean)<br/> 
Terminates in polynomial time (SLC) vs Terminates not in polynomial time (Kmean, EM)<br/> 


<h1 id="2">Module: UL3 - Feature Selection</h1>

**The Feature Selection Problem**:<br/>
Goals: 1) Knowledge Discovery, Interoperability & Insight. 2) Avoid Curse of Dimensionality, The amount of data you need grows exponentially with the number of features you have. By reducing our dimensions we can reduce the difficulty of the problem. In summary, with proper feature selection, not only will you understand your data better, but you will also have an easier learning problem<br/> 

Suppose we have N features and we want to reduce to M features where M <= N. How hard is this problem?<br/> 
Is it Polynomial (includes linear and quadratic) or exponential? ans: exponential<br/> 
To do this we need to come up with some sort of function that returns a score. We could choose M from N. It turns out this is a well known problem which NP-Hard. It's exactly because you have to find all possible subsets<br/> 


**Approaches to Feature Selection**:<br/>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul6.JPG?raw=true">
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
In fact Information Gain is often used in the search box. You could even go so far as using a decision tree (algorithm used in search box) to determine what should be passed to a Neural Network (actual learner, learner box). Basically, you are using the inductive bias of the decision tree to choose features, but then you use the inductive bias of your other learner in order to do learning. If the learner box is KNN, which suffers from the curse of dimensionality because it doesn't know which features are important, then this search box decision tree is good at figuring out which features are importent, and gives some hints to KNN.<br/> 


**Wrapping**: Given a set of features, apply a search algorithm to produce fewer features, pass them to the learning algorithm, then use the output to update the selected set of features. Here the ML quality is passed back as a quasi-score value which is then used by the algo to determine the final features. In summary, wrapping interacts with the learning algorithm directly to iteratively adjust the feature set <br/>
Pros: Takes into account model bias, score and learning  <br/>
Cons: Very slow <br/>
We can use different techniques to search: Local searching can be useful. We can use approaches directly from randomized optimization algorithms if we treat the learner's result as a fitness function. Other viable options include forward search and backward search. Note that we should avoid an exhaustive search here as this can make the problem under a worst case scenario<br/>
1. Randomized Optimization algorithms<br/>
2. Forward Selection: Select a feature and evaluate the effect of creating different combinations of this feature with other features. Stop once you stagnate. This is similar to Hill Climbing.<br/>
3. Backward Elimination: Start with all the features and evaluate the effect of eliminating each feature<br/>


**Describing Features**<br/>
When it comes to determining which features we should use, it’s necessary to differentiate them based on their viability in a general, statistical sense as well as in a learner-specific sense. We usually care more about usefulness, but relevance is what we generally use to get there.<br/>
**Relevance  ∼  Information**<br/>
**Usefulness  ∼  Error given a Model/learner**<br/>

Per the lecture, relevance measure effect on Bayesian Optimal Classifier. A feature can be strongly relevant if removing it degrade B.O.C. A feature can be weakly relevant if it is not strongly relevant, adding it to some subset of features S can improve B.O.C. Otherwise a feature is irrelevant. Usefulness measure effect on a particular predictor, minimizing error given a specific model/learner.<br/>
Think about how one would need at least two relevant features for decision to learn the AND function. For perceptron, it needs one more feature. That additional feature is useful to perceptron but irrelevant to decision tree. DT can learn the AND function without it. Perceptron cannot learn the AND function without it.<br/>

**Relevance**<br/>
Let B.O.C. = Bayes Optimal Classifier<br/>
• xi is strongly relevant if removing it degrades B.O.C.<br/>
• xi is weakly relevant if:<br/>
    1) Not strognly relavant <br/>
    2) ∃ a subset of features S, such that adding  xi  to S improves B.O.C.<br/>
• xi is otherwise irrelevant<br/>

**Usefulness**<br/>
• Relevance measures effect on B.O.C.<br/>
• Usefulness measures effect on a particular predictor<br/>


<h1 id="3">Module: UL4 - Feature Transformation</h1>

**What is Feature Transformation?**<br/>
The problem of pre-processing a set of features to create a new (smaller or more compact) feature set, while retaining as much (relevant and useful) information as possible.<br/>
Feature Selection is where the preprocessing is literally extracting a subset of the features. In Feature Transformation, we apply a “linear transformation operator”. The goal is to find a matrix 𝑃 such that we can project the examples into a new subspace (that is typically smaller than the original subspace) to get new features that are linear combinations of the old features.<br/>

<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul7.JPG?raw=true">
</p>

**Why Feature Transformation?**<br/>
Why we need linear transformation operator? We combine features together hoping to eliminate false positives/negatives<br/>

Motivation/information retrieval problem: <br/>
Given an unknown search query, we want to list documents from a massive database relevant to the query. How do we design this? If we treat words as features, we encounter the curse of dimensionality: there are a lot of words. Furthermore, words can be ambigious. Many contexts leading to different meanings (polysemy. Apple: fruit or the company?). Another challenge is synonomy: same meaning different representations (Car, Automobile, vehicle etc). Because of this, we encounter false positives and false negatives even if we could find the documents with the words efficiently. A good feature transformation will combine features together and provide a more compact way to query things<br/>


**Principal Components Analysis**<br/>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul9.JPG?raw=true">
</p>
An eigenproblem is a computational problem that can be solved by finding the eigenvalues and/or eigenvectors of a matrix. In PCA, we are analyzing the covariance matrix https://www.cs.princeton.edu/picasso/mats/PCA-Tutorial-Intuition_jp.pdf<br/> 
Principal Components Analysis is an example of an eigenproblem which will transform the features set by:<br/>
1) Finding the direction (vector) that maximizes variance. The is called the Principal Component<br/>
2) Finding directions that are mutually orthogonal to the Principal Component.<br/>
We are doing a transformation into a new space where feature selection can work<br/>

Properties:<br/>
1) global algorithm (will be forced to find global features): Mutually orthogonal means it is a global algorithm. "global" means that all the directions and new features that they find have a big global constraint, namely that they must be mutually orthogonal.<br/>
2) PCA gives the ability to do reconstruction, because it’s a linear rotation of the original space that minimizes L2 error by moving 𝑁 to 𝑀 dimensions. So, we don’t lose information. In details: the PCA actually gives you the best reconstruction, which means if I return these two dimensions, I have actually lost no information. It is just a linear rotation of the original dimensions. So if I were to give you back these two different features, you could reconstruct all of your original data. But PCA will take just one of these dimensions to reconstruct, in particular, take the first one as the principle component, I am guaranteed that if I project only into this space and then try to reproject into the original space, I will minimize the L2 error (The squared error, here the distance).What this means is that if I project onto this single axis here, and then I compare it to where it was in the original space, the distance, the sum of all the distances between those points will actually be the minimal that I could get for any other projection. just think about the fact that points. Always start out in some orthogonal space. And, I'm basically finding in scaling and a rotation such that I don't lose any information. And I maximize variance along the way. By maximizing variance, it turns out I'm maximizing or maintaining distances as best I can in any given dimension. that gives me the best reconstruction that I can imagine. <br/>
3) As an eigenproblem, each Principal Component has a prescribed eigenvalue. We can throw away the components with the least eigenvalues as they correspond to the features that matter less in the reconstruction. In details: What happens when you do principal components analysis is you get all of these axes back, and in fact, if you start out with N dimensions, you get back N dimensions again, and the job here for a future transformation as you might recall, is you want to pick a subset M of them hopefully much smaller than N. it turns out that associated with each one of these new dimensions that we get is its eigenvalue. That eigenvalue is guaranteed to be non-negative, it has a lot of other neat properties. But what matters to us here is that the **eigenvalues monotonically non-increase**, that is, they tend to get smaller as you move from the principal to the second principal, to the third, to the fourth, to the fifth, to the sixth, and so on to the nth dimension. And so, you can throw away the ones with the least eigenvalue. And that's a way of saying that you're throw awaying these projections, or the directions, or the features, with the least amount of variance.<br/>
4) If eigenvalue of some particular dimension equals to 0, then it means it provides no information (not irrelevant) in the original space. So throw away this dimension won't affect reconstruction.<br/> 

Pratical properties:<br/> 
5) It's well studied. In this case, it's very fast algorithms.<br/> 
6) Does it help with classification later? Maybe not. If one of the original dimension is directly related but its variance of that particular direction is extremely small. It might end up throwing it away. This doesn't help with classification later. kind like filtering. features with high variance don’t necessarily correlate to features with high importance.PCA will almost certainly drop the useful feature when the random noise has high variance<br/> 


**Independent Components Analysis**<br/>
ICA attempts to maximize independence. It tries to find a linear transformation of the feature space, such that each of the individual new features are mutually statistically independent:<br/> 
1) The mutual information between any two random features equals zero I(yi,yj)=0 <br/> 
2) The mutual information between the new features set and the old features set is as high as possible I(yi,xi)= maximum.<br/> 

In the other words, we want to be able to reconstruct the data (predict X from Y or Y from X). And at the same time, each variable in the new dimension is in fact mutually independent. <br/> 
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul10.JPG?raw=true">
</p>

Motivation: Blind source separation/cocktail party problem. This is the background for ICA: the voices are hidden variables (variables that we wish we knew since they powered the data we see) that we are trying to learn about, and the overall room noise is our known data. ICA reconstructs the sounds independently by modeling this assumption of the noise being a linear combination of some hidden variables<br/> 

PCA VS ICA<br/> 
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul11.JPG?raw=true">
</p>
Bag of Features means whether they both produce a set of feature vectors.<br/> 
By maximizing variance along orthogonal dimensions, PCA is finding uncorrelated dimensions. There are cases under which PCA happens to find independent projections when all data is gaussian, although independence is not the goal for PCA.<br/> 
The underlying model between PCA and ICA doesn't match. ICA should not maximizing variance. To do so will mix independent variables together (linear combination of independent variables) through central limit theorem to normal distribution, which in fact do not tease apart the independent things. This is specifically wrong when variables are highly non-normal distribution (ICA assumption) <br/>
PCA maximize reconstruction of the original data, not the mutal information<br/>
Effectively they're both trying to do the same thing (reconstruct the data) but they do it in very different ways. What both of these are great at is helping you to understand the fundamental structure and causes of our data.<br/> 
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul12.JPG?raw=true">
</p>

Directional means if give a matrix or a transpose of that matrix, PCA will end up finding the same answer. But ICA gives completely different answers. so ICA is highly directional. <br/><br/>


**Random Components Analysis/ Random Projection**<br/>
Similar to Principal Components Analysis, but instead of generating directions that maximize variance, it generates random directions to project data onto it. In doing so it still manages to pick up on some of the correlation.<br/>
It captures some of the correlations that works well with classification settings. Because you project it to lower dimension space that happens to capture some correlation. In some case the projected dimension m is even bigger than the original space n <br/>
It’s faster than PCA and ICA. It often won't reduce the dimensions as much as the other two approaches. So one big advantage of RCA is fast but it's also cheap and simple to implement.<br/>

**Linear Discriminant Analysis**<br/>
Linear Discriminant Analysis finds a projection that discriminates based on the label. That is, it finds projections of features that ultimately align best with the desired output. In other words, it aims to find a collection of good linear separators on data.<br/>
PCA, ICA and RCA feel more like filtering. But LDA is more similar to the wrapping function. LCA does care about the labels and wants to find ways to discrimination. Unlike wrapping, LDA does not care about the learner <br/>

<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ula1.JPG?raw=true">
</p>

<h1 id="4">Module: RL1 - Markov Decision Processes</h1>

**Decision Making & Reinforcement Learning** <br/>
Supervised Learning: given (𝑥, 𝑦) pairs, and the goal is to find a function 𝑓(𝑥) that maps a new 𝑥 to a proper 𝑦 (function approximation). <br/>
Unsupervised Learning: given a dataset of 𝑥 points, and the goal is to find a function 𝑓(𝑥) the provides a description of this dataset (Clustering). <br/>
Reinforcement Learning: a similar task as in Supervised Learning. But instead of having (𝑥, 𝑦) pairs as an input, we’ll be given (𝑥, 𝑧) pairs, and our job is to find 𝑓(𝑥) that generates 𝑦. <br/>


**Markov Decision Processes** <br/>
1) States S:  Set of tokens. The state of the object of interest represented in any way <br/>
2) Model T(s,a,s') ∼ Pr(s'|s,a): This is the Transition Model (function), which is the probability of transitioning to a new state 𝑠 given that the object starts at state 𝑠 and performs action a. 1) Markovian prop1 that only the present state matters, the past doesn't. 2) Markovian prop2 the transition model is stationary and doesn't change over time <br/>
3) Actions A(s)/A: the set of all possible actions<br/>
4) Reward R(s)/R(s,a)/R(s,a,s'): the reward the object gets when transitioning to a state 𝑠, which tells it the “usefulness” of transitioning to that state. It can be 1)R(s) is the reward received from being in a state 2)R(s,a) taking an action from a particular state 3)R(s,a,s') taking an action from a particular state and landing in state. They are all mathematically equivalent <br/>

The above is how we define the MDP problem, the policy is the solution that we are looking to determine.<br/>
Policy: π(s) → a.  π∗ is the optimal policy. The Policy says when you at s state, take action a. an agent always know what states it is in and what reward it can receive. The policy is a function 𝜋(𝑠) that takes a state 𝑠 as an input and returns the proper action 𝑎. A problem might have different policies. The optimum policy 𝜋∗ is the policy that maximizes the long term expected reward. RL talks about policies, a function that tells you what action to take for all possible states you are in. If you have a policy, and it is an optimal one, it will guide you in any situation. What the policy doesn't tell you is 2 or more steps into the future.<br/>
We will be given (𝑠, 𝑎, 𝑟) and our task would be to learn the optimum policy 𝜋∗ that produces the optimum action (highest reward).<br/>

**Rewards** <br/>
The amount of rewards? <br/>
1) In the Reinforcement Learning context, we don’t get a reward for each action. Instead, we get a “delayed reward” in terms of a positive or negative label for the state we end up in. <br/>
2) Because we don’t have instant rewards for each action, we have the problem of figuring out which specific action(s) lead us to a positive/negative outcome. This problem is called "Temporal Credit Assignment". <br/>
3) A small negative reward will encourage the agent to take the shortest path to the positive outcome. However, a big negative reward will encourage the agent to take the shortest path no matter what the result is, so it might end up with a negative outcome. A big positive reward will encourage the agent to never terminate the game. This means that determining the rewards is some sort of a domain knowledge. <br/>

How much time you have to reach the result? (Infinite horizon -> stationary)<br/>
The normal the gridworld probelm is an infinite horizon. If change horizon from infinite to finite, two things would happen:  1) policy changes because game may end 2) policy may chang even in the same state. For example, turn left and bounce the wall, stay, turn left and bounce wall, stay. Run out of time, turn right.<br/>
If you have an infinite amount of time (Infinite Horizon) you will be able to take longer paths to avoid possible risks. On the other hand, if you don’t have that much time, you’ll have to take the shortest path, even if it underlies some risk of falling into a negative outcome. <br/>
This means that the "time" will change the optimum policy as 𝜋(s,t) → 𝑎. Without assuming an Infinite Horizon, we will lose the notion of stationarity in our policies 𝜋(s) → 𝑎. <br/>

Utility of Sequences<br/>
<p align="center" width="100%">
    <img width="40%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul13.png?raw=true">
</p>

stationarity of preferences: if u(s0,s1,s2..)> u(s0,s'1,s'2...) then u(s1,s2..)> u(s'1,s'2...). Another way to state stationarity of preferences is that if I prefer a sequence of states today over another sequence of states, then I’d prefer that sequence of states over the same other sequence of states tomorrow.<br/>
This notion forces you to do some sort of reward addition, because nothing else will guarantee the stationary of preferences. So "𝑈(𝑠0,𝑠1,𝑠2,⋯)=∑𝑅(s_t) where t = 0 to inf" will be ture. If you don't do that, "if u(s0,s1,s2..)> u(s0,s'1,s'2...) then u(s1,s2..)> u(s'1,s'2...)" will not hold.<br/>
However, There is a problem regarding "𝑈(𝑠0,𝑠1,𝑠2,⋯)=∑𝑅(s_t) where t = 0 to inf". Consider two grid worlds: in G1 the rewards are steady at +1 per time step, and in G2 they alternate between +1, and +2. Which of the two is better? Well neither, because both sum to infinity when there is an infinite horizon<br/>

<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl1.png?raw=true">
</p>

To solve this problem by adding a variable in the function to get discounted reward: 𝑈(𝑠0,𝑠1,𝑠2,⋯)=∑ 𝛾^𝑡 * 𝑅(𝑠_𝑡) for 0<𝛾<1 <br/>
If 𝛾 → 0, we get the first reward and then everything else will fall off to nothing.<br/>
If 𝛾 → 1, we get a maximized reward<br/>
Now we can add an infinite number of numbers, and come up with a single number. Furthermore our intial assumption can still hold (Infinite Horizons, and Utility of sequences). To understand why the conclusion is true just solve for x using a geometric series<br/>


**Policy** <br/>
<p align="center" width="100%">
    <img width="40%" src="https://github.com/audrey617/Notes/blob/main/ML/images/ul14.png?raw=true">
</p>
Define  𝜋∗=𝑎𝑟𝑔𝑚𝑎𝑥_𝜋 𝐸(∑𝛾^𝑡𝑅(𝑠_𝑡)|𝜋] which is the policy that maximizes our long term rewards<br/>
𝑅(𝑠) ≠ 𝑈𝜋(𝑠). R(s) is immediate reward or feedback, whereas Utility is long term, or delayed, rewards.<br/>
the true utility of a state is defined by Bellman Equation: 𝑈(𝑠)=𝑅(𝑠)+𝛾⋅𝑚𝑎𝑥∑𝑇(𝑠,𝑎,𝑠′)𝑈(𝑠′) <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl2.png?raw=true">
</p>

**Solving Bellman Equation: Value Iteration and Policy Iteration** <br/>
If we have 𝑛 states/utilities, then we have 𝑛 equations in 𝑛 unknows. If the equations are linear, they would have been solvable, but the 𝑚𝑎𝑥 operator makes the equations non-linear. But there are still ways to solve it even it is non-linear<br/>

<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl3.png?raw=true">
</p>
Value Iteration:<br/>
steps<br/>
1. Start with arbitrary utilities.<br/>
2. Update utilities based on neighbors.<br/>
3. Repeat until convergence.<br/>
We basically update the estimate of utility of state 𝑠 by calculating the actual reward for this state plus the discounted utility expected from the original estimate of utility of s<br/>
This is guaranteed to converge because with each step we’re adding 𝑅(𝑠), which is a true value. So, even if we started with a very wrong estimate of utility, we keep adding the true value 𝑅(𝑠) in each iteration that it will dominate the original arbitrary estimate.<br/> 

<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl4.png?raw=true">
</p>
Policy Iteration:<br/>
steps<br/>
1. Start with an arbitrary policy 𝜋0<br/>
2. Evaluate: how good that policy is by calculating the utility with Bellman equation<br/>
3. Improve: 𝜋_𝑡+1 = argmax a based on the new utilities<br/>
Note that rather than having the 𝑚𝑎𝑥 over actions as in the normal Bellman equation, we already know what action to take according to the policy we’re evaluating. This trick removes the 𝑚𝑎𝑥 operator, making this a set of 𝑛 solvable linear equations in 𝑛 unknow<br/>

<h1 id="5">Module: RL2 - Reinforcement Learning</h1>

**RL API and Three approaches to RL** <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl5.png?raw=true">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl6.png?raw=true">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl7.png?raw=true">
</p>
In Markov Decision Process, our input is a model consisting of a transition function 𝑇 and a reward function 𝑅, and the intended output is to compute the policy 𝜋 (Planning).<br/>
In Reinforcement Learning, the inputs are transitions (Initial state, action, reward, result state, …), and the intended output is to “learn” the policy 𝜋. Reinforcement Learning is about “reward maximization”.<br/>
Three approaches:<br/>
1. Policy Search Algorithm: Mapping states to actions. Learning directly on policy for the use. However, learning this function is very indirect. Similar to Temporal Credit Assignment problem, state to action is not easy<br/>
2. Value Function based Algorithms: Mapping states to values. Learning values from states is quite direct, but turning this into a policy might be done using an 𝑎𝑟𝑔𝑚𝑎𝑥.<br/>
3. Model-based Algorithm: Mapping (states & actions) to (next state & reward). Turning this into a utility function can be done using Bellman equations, then using 𝑎𝑟𝑔𝑚𝑎𝑥 to get the policy. It is direct learning but the usage is computationally indirect.<br/>

**Q function - a new kind of value funciton** <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl8.png?raw=true">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl9.png?raw=true">
</p>

Q-function: the utility of leaving state 𝑠 via action 𝑎, which is the reward of state 𝑠 plus the discounted expected value of taking action 𝑎 multiplied by the value of the optimum action in state 𝑠′. 𝑈(𝑠) and 𝜋(𝑠) can be defined via 𝑄(𝑠, 𝑎) in the chart. Estimating the value of 𝑄(𝑠, 𝑎) or evaluating hte bellman equations from data is called Q-Learning<br/>

**Q Learning** <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl10.png?raw=true">
</p>
Q-Learning is estimating the value of 𝑄(𝑠, 𝑎) based on transitions and rewards but we don't have access to 𝑅(𝑠) and 𝑇(𝑠, 𝑎, 𝑠′)<br/>
𝑄̂(𝑠, 𝑎) is an estimate of the Q-function that updates by a learning rate 𝛼 in the direction of the immediate reward 𝑟 plus the estimated value of the next state. 𝛼 is 0 corresponds to no learning. nothing will change. 𝛼 is 1 means full learning. we forget what we learnt before and jump into the new value. <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl11.png?raw=true">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl12.png?raw=true">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl13.png?raw=true">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/rl14.png?raw=true">
</p>








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
game theory check George Kudrayvtsev's note
