# Lesson outline
- [Module: ML is the ROX](#1)
- [Module: SL1  Decision Tree](#2)
- [Module: SL2  Regression & Classification](#3)
- [Module: SL3  Neural Networks](#4)
- [Module: SL4  Instance Based Learning](#5)
- [Module: SL5  Ensemble B&B](#6)
- [Module: SL6  Kernel Methods & SVMs](#7)
- [Module: SL7  Comp Learning Theory](#8)
- [Module: SL8  VC Dimensions](#9)
- [Module: SL9  Bayesian Learning](#10)
- [Module: SL10 Bayesian Inference](#11)
- ----------------------------------------------
- [Module: UL5 - Info Theory](#12)
- [Module: UL1 - Randomized Optimization](#13)


 
<h1 id="1">Module: ML is the ROX</h1>

## 1. Supervised Learning VS Unsupervised learning VS Reinforcement Learning
Supervised Learning = Approximation. Inductive bias.<br />
Unsupervised learning = Description<br />
Reinforcement Learning (Wiki): <br />
How intelligent agents ought to take actions in an environment in order to maximize the notion of cumulative reward<br />

For optimization:<br />
supervised learning - label data well to find the function/ approximate function<br />
unsupervised learning - make up some criterion and then find a way to cluster data to scores well.<br />
reinforcement learning - try to find behavior that scores well<br />

## 2. Induction and Deduction
Induction: being from examples to a more general rule.<br />
Deduction: from a general rule to specific examples, like reasoning.<br />
inductive bias: <br />
https://www.baeldung.com/cs/ml-inductive-bias<br />
preference and restriction biases for supervised learning algorithms:<br />
https://downey.io/notes/omscs/cs7641/restriction-and-preference-bias-supervised-learning/<br />

Data, Learning and Modeling<br />
https://machinelearningmastery.com/data-learning-and-modeling/#:~:text=Instance%3A%20A%20single%20row%20of,attribute%20of%20a%20data%20instance.


<h1 id="2">Module: SL1 Decision Tree</h1>

## 1. Terms definition
 * Instances: inputs, a row of data, an observation from domain <br />
 * Concept: A function mapping inputs to outputs<br />
 * Target concept: The actual answer/function we try to find<br />
 * Hypothesis class: The set of all concepts. All functions I am willing to think about. Could be all possible functions  * however that would be hard for us to figure out which one is the right one given finite data.<br />
 * Sample (Training set): A set of instances with correct labels<br />
 * Candidate: A concept that you think might be the target concept<br />
 * Testing set: A set of instances with correct labels that was not visible to the learning algorithm<br />

## 2. Representation VS Algorithm
 * DT Representation: Represent data. Decision tree is a strcuture of nodes(condition/feature), edges(feature value, eg T/F) and leaves (output label)<br/>
 * DT Algorithm: Sequence of steps to get output 1) Pick best attribute 2) Ask question 3) Follow answer path 4) repeat 1-3 until find answer<br/>

## 3. Decision Trees Expressiveness
For Boolean function (AND & OR), the truth table is a path to a leaf.<br/>
A AND B: A->T->B->T-> True leaf, otherwise False leaf<br/>
A OR B : A->F->B->F-> FALSE leaf, otherwise True leaf<br/>
<br />
For Boolean function (XOR)<br/>
The tree is just another representation of the full truth table. <br/>
<br/>
A generalized situation<br />
N-OR (A1 OR A2 OR A3 OR A4..., probelm like any): We need N nodes to represent tree. The decision tree is in fact linear.<br />
N-XOR  (A1 XOR A2 XOR A3 XOR A4...problem like parity (odd parity)): We need 2^N - 1 or O(2^N) nodes to represent tree. The decision tree is exponential number of nodes. N-XOR is exponential and hard.<br />
<br />
We prefer any than parity. If we do sum instead of odd parity, then we transfer this N-XOR to a easier problem. So a hard problem is to come up a good representation, or find a way to cheat.<br />
<br />
- - - -
When we have N boolean attributes (will need n! nodes), output is boolean. <br />
 * How many rows will we have for truth table itself - > 2^N<br />
 * How many decision tree (functions) will we have? or In terms of n, how many different ways might we fill out this column of binary outputs? -> 2^(2^n) <br />
 * Explain from professor: Well if you understand how we got 2^n from the possible patterns of n attributes, then going to to the next step is exactly the same reasoning. With n attributes there are 2^n possible patterns. Let's write that k patterns so I can stop using carets all the time. So there are k possible patterns. I want to know how many binary functions there are over k patterns.  How do we figure that out? Well each pattern can be mapped to true or false. So that means each pattern can be labeled one of two ways. Therefore there 2^k ways to map those patterns (dammit, the caret is back). What's k? Oh, 2^n. So there are 2^(2^n) boolean functions from n boolean attributes.<br />
 * This is double exponential. 2^(2^n) is not 4^n. This is a lot. When N is 6, the number is already very big.<br />
 * So the space of decision tree is large. The hypothesis space of all decision trees is very expressive and expensive if we don't make smart decision to find the right tree.

<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/1.JPG?raw=true">
</p>

### 4. ID3 (Top down, greedy approach, returns optimal decision tree, prefer shorter tree than long tree)
The leaves contrains a mixture of T and F are impure. The leaves only contain T or F are pure. To select node, compare options, we prefer the option provides more pure leaves. To quantify the impurity of leaves, we use Gini impurity (G= ∑ p(i)∗(1−p(i)). A Gini Impurity of 0 is the lowest and best possible impurity), entropy or Information gain. <br/> 
<br/> 
Best selection of ID3 is based on **largest Information gain or smallest entropy**<br />
IG = H(S) - H(S|A). H(S|A) is uncertainty(entropy) after splitting set S  on attribute A.<br />
https://en.wikipedia.org/wiki/ID3_algorithm  & StatQuest 

<p align="center" width="100%">
    <img width="30%" src="https://github.com/audrey617/Notes/blob/main/ML/images/3.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/4.JPG?raw=true">
</p>

```
ID3 (Examples, Target_Attribute, Attributes)
    Create a root node for the tree
    If all examples are positive, Return the single-node tree Root, with label = +.
    If all examples are negative, Return the single-node tree Root, with label = -.
    If number of predicting attributes is empty, then Return the single node tree Root,
    with label = most common value of the target attribute in the examples.
    Otherwise Begin
        A ← The Attribute that best classifies examples.
        Decision Tree attribute for Root = A.
        For each possible value, vi, of A,
            Add a new tree branch below Root, corresponding to the test A = vi.
            Let Examples(vi) be the subset of examples that have the value vi for A
            If Examples(vi) is empty
                Then below this new branch add a leaf node with label = most common target value in the examples
            Else below this new branch add the subtree ID3 (Examples(vi), Target_Attribute, Attributes – {A})
    End
    Return Root
```

### 5. ID3 Bias (Inductive Bias)
Restriction Bias: hypothesis set space H. <br/>
Reference Bias: subset of hypothesis (n belongs to H). Short or long, how to split(gini or entropy), which tree to prefer (accuracy? precision?)<br/>

### 5. Other considerations(Continuous Attributes, Repeat attribute, When to stop, regression tree)
**Continuous Attributes**: split attribute range into equal intervals<br/>
**Repeat attribute along a path in a tree**: NO for discrete attributes But YES for continuous attributes since we can ask different question on the same attribute. eg, ask age attribute "is it above 30", then ask "is it above 15" makes sense.<br/> 
**When to stop**: <br/> 
1.Everything is cliassified correctly <br/> 
2.No more attributes <br/> 
3.overfitting happens: <br/> 
1) cross-validation,<br/>
2) validation curve& learning curve,<br/>
3) pre-pruning or post-pruning. post-pruning: Cost complexity pruning, essentially, pruning recursively finds the node with the “weakest link.” The weakest link is characterized by an effective alpha, where the nodes with the smallest effective alpha are pruned first. cost complexity measure/tree score = Training error + a * T (number of leaf nodes).  The a * T is the tree complexity penalty. a is the tuning value. We picked the sub tree with lowest tree score. a = 0, original full tree <br/> 
4) Don't violate Occam's razor: entities should not be multiplied beyond necessity<br/> 
<br/> 

**Regression Tree**<br/> 
In a regression tree, each leaf represents a numeric value. In contrast, classification tree has either true or false in leaves or the leaves are discrete categories.<br/> 
To pick one feature's best threshold to split data into two groups, we try to find the threshold with the smallest sum of squared residuals.<br/> 
To build a tree, From root, we have each feature pick its best threshold, which becomes a candidate for the node. We compare each candidate's SSRs, and then pick the candidate with the lowest value for root. We grow the tree in this way<br/> 
<br/> 
What to do for splitting: Need continuous outputs. Information gain is not available since it cannot measure information on continuous values well and won't generalize well. But we can visualize how bad a prediction is by looking at the distance between the observation and predicted values. This distance is residual. And we can use the residuals to quantify the quality of these predictions. To evaluate the prediction of the threshold selection, we add the squared residuals of each sample as the sum of squared residuals. Measure errors/mixedup things can also use variance. Gain ratio is also one option. <br/> 
What to do for leaves:  Average, local linear fit.<br/> 



<h1 id="3">Module: SL2 Regression & Classification</h1>
Regression: falling back to mean <br/> 
Linear Regression (Traditional Statistic):  <br/> 
1) Use least-sqaures to fit a line to data  <br/> 
2) Calculate R^2(coefficient of determination, (SS(mean)-SS(fit))/SS(mean)) which describes how well the regression predictions approximate the real data points   <br/> 
3) Calculate a p-value for R^2. Imagine the data only has two observations, R^2 will be 100% as long as you draw a striaight line. We need more information to determine if the R^2 is statistically significant or reliable. This is p-value. The p-value for R^2 comes from F=((SS(mean)-SS(fit))/degree of freedom pfit-pmean)/(SS(fit)/degree of freedom n-pfit). The p-value is number of extreme values divided by all values <br/>
4) https://en.wikipedia.org/wiki/Regression_analysis <br/>

### 1. Errors
Our goal is to find the values of θ(coefficient) that minimize the above sum of squared errors (Mean Sqaure error. MSE). One of the common approach is to use calculus. Another approach is where the gradient descent algorithm comes in handy. Also notice, how easy it is to take a derivative of this error function. So take a good look at the gradient descent algorithm document and come back here to find the linear equation that fits our data.<br/>

### 2. Polynomial Regression
General linear model. Detail see wiki link. Get weight/coefficient <br/> 

$$ W = (X^TX)^{-1}X^TY $$  

### 3. Model Selection & overfitting/underfitting Cross Validation
The goal of Machine Learning is "Generalization". One meaning of "fold": "consisting of so many parts or facets." So, n-fold cross validation means the data is in n parts. - Michael Littman <br/>
Overfitting, underfitting - learning curve & validation curve <br/> 
https://scikit-learn.org/stable/modules/cross_validation.html <br/> 
https://en.wikipedia.org/wiki/Cross-validation_(statistics) <br/> 

### 4. Input Spaces
Scalar continuous input  <br/> 
vector continuous input  <br/> 
discrete input, Scalar or vector<br/> 

<h1 id="4">Module: SL3 Neural Networks</h1>

### 1. Perceptron

A perceptron is a linear function (equal to threshold), and it computes hyperplanes.<br/> 

**Perceptron units expressions of boolean** <br/> 
If we focus on X1 ∈ {0,1} and X2 ∈ {0,1}. What W1,W2 and θ can be?<br/> 
AND: 1/2, 1/2, 3/4<br/> 
OR: 1/2, 1/2, 1/4<br/> 
NOT for X1: W1 = -1, θ = 0<br/> 
XOR: requires 2 perceptrons<br/> 
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/5.JPG?raw=true">
</p>

**Perceptron Training** <br/> 
Given examples, find weights that map inputs to outputs. Two different rules are developed. One is Perceptron rule (use threshold output) and the other is gradient descent/delta rule (use unthreshold values)<br/> 

<strong>Perceptron Training: Perceptron rule. Δw_i = α * (y - ŷ)*x_i. Finite convergency for linear separability</strong><br/> 
https://en.wikipedia.org/wiki/Perceptron See learning algorithm part for details.<br/> 
The idea is to add weights when y=1 and ŷ = 0 and to reduce weights when y=0 and ŷ = 1. Learning rate is used to control the weight change speed so as to avoid overshooting. If the data is linearly separable, the perceptron will find the seperate line in finite iterations. However, whether a data is linearly separable is usualy unknown. So we use threshold to stop loop: repeated until the iteration error is less than a user-specified error threshold. (If it is known this dataset is linear seperatable, we could set the error to 0. But maybe not ideal to do so), or a predetermined number of iterations have been completed then stop.
<br/> 
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/6.JPG?raw=true">
</p>

<strong>Perceptron Training: Gradient descent. Δw_i = α * (y - a)*x_i. More robust for non-linear. Local optimum if not convex</strong><br/> 
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/7.JPG?raw=true">
</p>

```
Take the derivative of the Loss function for each parameter in it
Pick random values for parameters
while(stepsize very small or reach max number of steps){
    Plug the parameter values in to the derivatives (Gradient) 
    Calculate the step size. stepsize = slope * learning rate
    Calculate new parameters. New parameter = old parameter - step size 
}
```
<br/> 

### 2. Sigmoid (S-like, Differentiable threshold)
Similarity between the above two functions begs the question, why didn’t we just use calculus on the thresholded ŷ? The simple answer is that the function ŷ is not differentiable (https://en.wikipedia.org/wiki/Differentiable_function) <br/> 
How differentiable? sigmoid is one option. Perceptron is a "hard" version of sigmoid function. When a-> -inf, sigmoid(a)->0, when a->+inf, sigmoid(a)->1 <br/>
From George Kudrayvtsev student note<br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/8_a.JPG?raw=true">
</p>


Regarding activation functions in NN <br/>
https://towardsdatascience.com/activation-functions-neural-networks-1cbd9f8d91d6 <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/activation_function_cheatsheet.png?raw=true">
</p>

### 3. Neural Network
**Sketch** <br/> 
When activation function is differentiable like sigmoid, then mapping from input to output will be differentiable in terms of weights, which means we can figure out how any given weight change in the network changes the mapping from inputs to outputs. This leads to backpropagation (information flows from input to output and error flows backward from output to input. This tells you how to compute derivatives)<br/> 
Backpropagation: the error of the network propogates to adjust each unit’s weight individually.<br/> 
We don't have guarantee of convergency in finite time. No hard thresholding<br/>
It could be stuck in local optimal<br/>

**Optimizing Weights** <br/> 
Gradient descent can get stuck in local optima and not necessarily result in the best global approximation of the function in question. Besides gradient descent, other methods to train NN <br/>
1) Momentum: allows gradient descent to “gain speed” if it’s descending down steep areas in the function <br/>
2) Higher order derivatives: look at combinations of weight changes to try to grasp the bigger picture of how the function is changing <br/>
3) Randomized optimization<br/>
4) Penalizing complexity: the idea of penalizing “complexity” so that the network avoids overfitting with too many nodes or too many layer or too large magnitude of weights <br/>


**Restriction Bias** <br/>
Restriction bias tells you something about the representational power of whatever data structure you use, in this case, the network of neurons. And it tells you the set of hypotheses that you are willing to consider. It is the representation's ability to consider hypotheses <br/>
<br/>
What restriction we are putting? Perception can only work with linear data or half spaces, but NN restriction bias is not much restricted if using sigmoids. NN can model many types of functions: <br/>
1) Boolean: Network of threshold-like unit
2) Continuous Functions (a function with no jump or discontinuities): represented with a single hidden layer with enough hidden units. Each hidden unit can worry about one little patch of the function that it needs to model. The patch got set in the hidden layer and in the output layer they get stitched together.
3) Arbitrary: Anything, even continuous has discontinuities. The solution is to add hidden layers. With multiple hidden layers, it works<br/>

NN has low restriction bias but high probability of overfitting due to model complexity and excessive trainig. To avoid that, we restrict to a bounded number of hidden layers with bounded number of units and stop training when weights are too large . The number can be decided using cross validation. Error on the training set drops as we increase iteration but will cause overfit in the end <br/>


**Preference Bias** <br/>
Preference bias describe which hypotheses from the restricted space are preferred. Give two representation, Why would you prefer one over the other. <br/>
How do initial the weights: small random values. random help avoid local minima; small help avoid overfitting (too large magnitude of weights->overfitting); small and random has low complexity -> Meet Occam's razor (Don't make something more complex unless you are getting better error; if two things have similar error, pick simpler one for generalization)<br/>


### 4.Neural Network From StatQuest
**Part1: Inside BlackBox** <br/>
A neural network consists of Nodes and connection between the nodes. The numbers along each connection represent parameter values (weights and biases) that were estimated when this NN was fit to the data. It starts out with unknown values that are estimated when we fit NN to a datase using Backpropagation. Usually a neural network has more than one input/output node and different layers of nodes between input and output nodes. The layers of nodes between the input and output nodes are called hidden layers. When you build a neural network, one of the first thing you do is decide how many hidden layers you want, and how many nodes go into each hidden layer. The hidden layer nodes contains activation functions/curved bent lines. The previous layer node output with the (weights * x + biases) becomes the input in the new layer node activation function. The node has the same activation function, but the weights and biases on the connection slice them, flip and stretch them into new shapes. In the end they get added with parameter adjustment, so we get a new squiggle green line for final prediction. <br/>

<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition1.JPG?raw=true">
</p>

**Part2: Backpropagation Main Ideas** <br/>
Step1: using chain rule to calculate derivatives  <br/> 
Step2: plug the derivates into Gradient Descent to optimize parameters <br/> 

<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_0.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_1.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_2.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_3.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_4.JPG?raw=true">
</p>



<h1 id="5">Module: SL4 Instance Based Learning</h1>

### 1.Instance Based Learning
Normal ML algorithms uses input data (𝑥, 𝑦) and searches the hypotheses space for the best generalized function 𝑓(𝑥) to predict new values. In Instance Based Learning, we create a database of all 𝑥/𝑦 relationships, and once we receive a new value 𝑥 we lookup this database to find corresponding 𝑦.<br/>
<br/>
Advantages: 1) The model perfectly remembers the training data rather than an abstract generalizing 2) Fast. No need for learning 3)simple <br/>
Disadvantages: 1) Massive storage to query 2) No generalization and overfitting: sensitive to noise 3) Can return multiple values for the same input <br/>

### 2.KNN(K-Nearest Neighbors)
**Algorithm** <br/>
While k is the number to consider, we also need “distance” to determine how close or similar an xi ∈ X is for a new input x. The distance is our expression of domain knowledge about the space
<br/>
```
Given:
    1. Training data D = {X,Y}
    2. Distance metric 𝑑(𝑞, 𝑥) → similarity function, domain knowledge
    3. Number of neighbors (𝑘) → domain knowledge
    4. Query point (𝑞)
    
Find:
    A set of nearest neighbors such that 𝑑(𝑞, 𝑥) is smallest

Return:
     1. Classification: vote, take the mode or plurality. 
     2. Regression: mean
     Tie needs tiebreak (random pick, closest distance). 
     Can also use a weighted vote of weighted avg=> the closer the point is, the more influence it has on the vote/mean
```

<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/10.JPG?raw=true">
</p>


**Running Time and Space Comparison Given n sorted data points** <br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/9.JPG?raw=true">
</p>


**Eager vs Lazy Learners Comparison** <br/>
https://ibug.doc.ic.ac.uk/media/uploads/documents/courses/ml-lecture4.pdf<br/>
https://jmvidal.cse.sc.edu/talks/instancelearning/lazyandeagerlearning.html<br/>
<br/>
1. Generalize at When: Instance-based methods are also known as lazy learning because they do not generalize until needed. All the other learning methods we have seen (and even radial basis function networks) are eager learning methods because they generalize (one-fits-all) before seeing the query. <br/>
2. Approximation: The eager learner must create a global approximation. The lazy learner can create many local approximations. <br/>
3. Performance: Lazy learning is very suitable for complex and incomplete problem domains, where a complex target function can be represented by a collection of less complex local approximations.If eager and lazy learners both use the same Hypothesis then, in effect, the lazy can represent more complex function. For example, if Hypothesis consists of linear function then a lazy learner can construct what amounts to a non-linear global function.<br/>
<br/>

**About Lazy Learners/Instance Based Learning** <br/>
Lazy Learners includes KNN, Locally weighted regression, Case-based reasoning <br/>
<br/>
Advantages: Incremental (online) learning, Suitability for complex and incomplete problem domains, Suitability for simultaneous application to multiple problems, Ease of maintenance <br/>
<br/>
Disadvantages: Handling very large problem domains, Handling highly dynamic problem domains, Handling overly noisy data, Achieving fully automatic operation (Only for complete problem domains a fully automatic operation of a lazy learner can be expected. Otherwise, user feedback is needed for situations for which the learner has no solution) <br/>
 

**KNN BIAS** <br/>
Restriction bias<br/>
Nonparametric regression: should be able to model anything as long as you can find a way to compute distance (similarity) between neighbors<br/>
<br/>

Preference Bias (Bias in Assumption. Our belief about what makes a good hypothesis): <br/>
1. locality(Near points are similar) -> d() distance function (euclidean, manhattan,...)<br/>
2. Smoothness -> k and avg. averaging neighbors makes sense and feature behavior smoothly transitions between values<br/>
3. Treating the features of training sample vector equally -> But is this really true? You may care more about x1 and x2 is less crucial. <br/>


**Curse of Dimensionality**<br/>
As the number of features or dimensions grows, the amount of data that we need to generalize accurately grows exponentially.
This is not only for knn but general ML<br/>
<br/>
From wiki: https://en.wikipedia.org/wiki/Curse_of_dimensionality<br/>
The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces that do not occur in low-dimensional settings such as the three-dimensional physical space of everyday experience. The expression was coined by Richard E. Bellman when considering problems in dynamic programming.<br/><br/>
Dimensionally cursed phenomena occur in domains such as numerical analysis, sampling, combinatorics, machine learning, data mining and databases. The common theme of these problems is that when the dimensionality increases, the volume of the space increases so fast that the available data become sparse. In order to obtain a reliable result, the amount of data needed often grows exponentially with the dimensionality. Also, organizing and searching data often relies on detecting areas where objects form groups with similar properties; in high dimensional data, however, all objects appear to be sparse and dissimilar in many ways, which prevents common data organization strategies from being efficient.<br/>
<br/>
As dimensions increase, What will happen? From JPL speech:<br/>
1) Euclidean distances become less meaningful<br/>
2) Uniform distributions become exponentially hard to sample<br/>
3) Many parameters become polynomially hard to estimate (eg, covariance)<br/>
4) Data becomes more difficult to visualize<br/>


**About d() and k**<br/>
How to pick d()? d(x,q) = euclidean/manhattan/weighted. Your choice of distance function really matters. The weighted distance is actually one way to deal with the curse of dimensionality. <br/>

How to pick k? When k is small, models have high variance, fitting on a strongly local level. Larger k creates models with lower variance but higher bias, smoothing out the influence of individual data points on output decisions. If k=n we will end up with a constant average function (under non-weighted average case). If we used weighted average, the points closer to the query point will have greater influence. We can also use other methods instead of weighted-average. We can use "local" linear regression on the nearest k points, this is known as "Locally Weighted Regression". We can even use Decision Trees, Neural Networks, etc. to get more complicated hypothese spaces (From line to curves for example). This though might cause overfitting<br/>

### 3.No free lunch theorem
Any learning algorithm that you create is going to have the property that if you average over all possible instance, it is not doing any different than random. If I don't know the data I am going to learn over, then it really doesn't matter what I do because there are all possible kind of dataset. However, If I have the domain knowledge, I can use it to choose the best learning algorithm for the problems that I am going to encounter <br/>
<br/>
From https://machinelearningmastery.com/no-free-lunch-theorem-for-machine-learning/ <br/>
The NFL stated that within certain constraints, over the space of all possible problems, every optimization technique will perform as well as every other one on average (including Random Search). If one algorithm performs better than another algorithm on one class of problems, then it will perform worse on another class of problems <br/>

<h1 id="6">Module: SL5  Ensemble B&B</h1>

### 1. Ensemble learning
The general approach to ensemble learning algorithms is to learn rules over smaller subsets of the training data, then combine all of the rules into a collective, smarter decision-maker. A particular rule might apply well to a subset, but might not be as prevalent in the whole; hence, each weak learner picks up simple rules that, when combined with the other learners, can make more-complex inferences about the overall dataset. We need to determine how to pick subsets (eg, Uniformly Randomly) and how to combine learners (eg, equally believe each one and take average).<br/>


### 2. Bagging/Bootstrap Aggregating (Bagged Decision Trees (canonical bagging), Random Forest, Extra Trees)
Each subsets are like boots and the averaging is like the strap. <br/>
**How to pick subsets**:  choosing data uniformally randomly to form our subset. Note, we can randomly select the same value more than once. Randomly selecting data and allowing for duplicates is called **Sampling with Replacement** <br/>
**How to combine learners**: combining the results with mean <br/>
Generalization: Taking the average of a set of weak learners trained on subsets of the data can outperform a single learner trained on the entire dataset because of overfitting. Overfitting a subset will not overfit the overall dataset, and the average will “smooth out” the specifics of each individual learner.<br/>


### 3. Boosting (AdaBoost (canonical boosting), Gradient Boosting Machines, Stochastic Gradient Boosting (XGBoost and similar))
The term boosting refers to a family of algorithms that are able to convert weak learners to strong learners. Boosting is an ensemble method that seeks to change the training data to focus attention on examples that previous fit models on the training dataset have gotten wrong. The key property of boosting ensembles is the idea of correcting prediction errors. The models are fit and added to the ensemble sequentially such that the second model attempts to correct the predictions of the first model, the third corrects the second model, and so on. From https://machinelearningmastery.com/tour-of-ensemble-learning-algorithms/<br/>

**3.1 How to pick subsets**: focus on subsets with "hardest" examples (Not good at analyzing)

**3.2 How to combine learners**: Weighted mean

**3.3 Error**: Learning only happens if your training set has the same distribution as the future testing set. If not, all bets are off. Here is another definition of error 𝑃𝑟_𝔻(h(𝑥) ≠ c(𝑥)). 𝔻 stands for distribution. h is the specific hypothesis that our learner think is the true concept. c is the underlying true concept. Now the error is the probability given the underlined distribution that I will disagree with the true concept on some particular instance X. <br/>

Why we consider this way? Is it same as considering only the number of mismatches in classification case? <br/>
No. Even you may get many examples wrong, in some sense, some examples are more important than others as some may be very rare.  It's not about the number of distinct mistakes you can make but rather the amount of time you will be wrong. This becomes important when you think about the underlying distribution of examples <br/>

**3.4 Weak Learner**: No matter what the distribution is over data, a learner will do better than chance (better than chance: the error rate Pr_𝔻(.) is always less than a half). ∀𝔻 : Pr_𝔻(.) ≤ 1/2 − ε. ε here means a very small number. Technically you are bounded away from one half. Another way to think about that is, you always get some information from learner. The learner is always able to learn something. Chance would be the case where your probability is 1/2 and you actually learn nothing at all.<br/>

Strong Learners vs. Weak Learners (The Strength of Weak Learnability, 1990):<br/>
A weak learner produces a classifier which is only slightly more accurate than random classification.<br/>
A class of concepts is learnable (or strongly learnable) if there exists a polynomial-time algorithm that achieves low error with high confidence for all concepts in the class.<br/> 

**3.5 Boosting In Code**: 
High level: Look at training data, construct distribution, find a weak classfier with low error. Keep doing that untile you have a bunch of them. And combine them somehow into some final hypothesis. But where to find the distribution and where do we get this final hypothesis?<br/> 

```
Given training data {(x_i, y_i)}, y_i ∈ {−1, +1}
For t (timestamp) = 1 to T:
    1) Construct the distribution 𝔻_t
    2) Find weak classifier H𝑡(𝑥) with small error ε = 𝑃𝑟_𝔻(h(𝑥) ≠ c(𝑥)) (ε could be as big as slightly less than a half)
Output H_final
```

**3.6 AdaBoost Math**: <br/> 
The better we’re doing overall, the more we should focus on individual mistakes. On each iteration, our probability distribution adjusts to make D favor incorrect answers so that our classifier H can learn them better on the next round; it weighs incorrect results more and more as the overall model performance increases. <br/> 

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/Add1.JPG?raw=true">
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/adaboost1.jfif?raw=true">
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/Add2.JPG?raw=true">
</p>

**3.6 AdaBoost Intuition**:  <br/> 
Intuition: why does boosting do well?  <br/> 
Boosting basically says if I have some examples that I haven't been able to classify well, I am going to rerate all my examples, so the one I'm not doing well become critical important. This is whole bit of 𝔻 all about. We know we have a notion of a weak leaner ht(x). No matter what happens, we will find some hypothesis that does well.  So in the end, to understand why the final hypothesis does well, we can ask under what circumstance it wouldn't do well. If it doesn't do well, there has to be a bunch of examples getting wrong. How many things could it not get right? That number has to be small. Let's imagine I had a number of examples wrong at a timestamp, since I have a distribution and I re-normalize, and then I get half right in the next round. It is forced to do well due to weak learner. This cause us to get fewer things wrong over time.  Can the right examples become wrong after re-distribution? (I am a bit confused on the proof.)  <br/> 

Boosting doesn't overfit <br/> 


### 4.  Adaboost from StatQuest
Using Decision Trees and Random Forests to explain the three main concepts behind AdaBoost.<br/> 
Comparison 1: In a Random Forest, each time you make a tree, you make a full sized tree. Some trees might be bigger than others, but there is no predetermined maximum depth. In contrast, in a Forest of Trees(Stumps) with AdaBoost, the trees are usually just a node and two leaves. A tree with just one node and two leaves is called a stump. Stump can only use one variable to make a decision, thus, Stumps are technically "weak learners". However, that's the way AdaBoost likes it.<br/> 
Comparison 2: In a Random Forest, each tree has an equal vote on the final classification. In contrast, in a Forest of Stumps made with AdaBoost, some stumps get more say in the final classification than others. Large stumps weight better than smaller stumps.<br/> 
Comparison 3: In a Random Forest, each tree is made independently of the others. In other words, it doesn't matter if one tree is made first than other. In contrast, in a Forest of Stumps made with AdaBoost, order is important. The errosrs that the first stump makes influence how the second stump is made.  The second stump influece the third and so on. <br/> 

In summary, AdaBoost 1) combines a lot of weak learner to make classficiations. The weak learner are almost always stumps. 2) Some stumps get more say in the classfication than others. 3) Each Stump is made by taking the previous stump's mistakes into account. <br/>

<br/>

<h1 id="7">Module: SL6  Kernel Methods & SVMs</h1>

### 1. Support Vector Machines (SVMs)<br/>
**1.1 About the margin** <br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVM1.JPG?raw=true">
</p>


<br/>**1.2 Max margin math transform**<br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVM2.JPG?raw=true">
</p>


**1.3 Properties of this 𝑊(∝) equaltion:**<br/>
1. Once you find alpha, you can recover W. W = ∑ a_𝑖 𝑦_𝑖 𝑥_𝑖  => Once you recover W, you can have b<br/>
2. Most a_i are going to be 0. This measn buch of the data don't really factor into W. So you basically build machine with a few support vectors (a_𝑖 are not 0). **Only few X_i matters**. The points that are far away from decision boundary and cannot be used to define the contours of that deicison boundary don't matter, whether they are plus or minus. <br/>
3. It's like **KNN** except you already done the work of figuring out which points actually matter. So you don't need to keep all of them. It doesn't just take the nearest ones but actually does this complicated quadratic program to figure out which ones are actually going to contribute. It's another way to think about instance-based learning, except that rather than being completely lazy, you put some energy into figuring out which points you could stand to throw away <br/>
4. What does the Xi transpose Xj (𝑥^𝑇𝑥) actaully means in 𝑊(a) ?  It's dot product, which like the projection of one of those onto the other. it ends give you a number, which is the length of projection indicating how much they are pointing in the same direction. It's a measure of their similarity. So Xi transpose Xj is a notion of **similarity**.<br/><br/>

Look back at 𝑊(a), it basically tries to
1) find all pairs of points
2) figure out which ones matter for finding your decision boundry (a)
3) think about how they relate to one another in term of their output labels (yy) with respect how similar (xTx) they are to each other.<br/><br/>


**1.4 Linearly Married**<br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVM3.JPG?raw=true">
</p>

This is a cute trick. It takes data nad transform it into a higher dimensional space where suddenly I am able to separate it linearly. It not only fits the circle pattern, but also doesn't require that I do this particuilar transformation. I can simply compute the dot product. In this formulation of the quadratic program 𝑊(∝), if you write code to do that, each time in the code you want to compute that Xi transpose Xj (𝑥^𝑇𝑥), if you just squared it right before you use it, it would be as if you projected it into this third dimension and found a plane. This is called the **kernel trick**<br/>
We care about maximizing some function 𝑊(∝) that depends highly upon how differnt data poins are alike defined by inner product 𝑥^𝑇𝑥. But instead, we can define similarity notion as (𝑥^𝑇𝑥)^2. In fact, we actually never use ϕ(q) = <q1^2,q2^2,sqrt(2)q1q2>. That just so happened to represent something in a higher dimensional space<br/>
It turns out for any function that you use, there is some transformation into some higher dimensional space, that is equivalent <br/>

**1.5 Kernel**<br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVM4.JPG?raw=true">
</p>


**Predefined kernels** <br/>
Below From https://towardsdatascience.com/an-intro-to-kernels-9ff6c6a6a8dc<br/>
The problem with mapping data onto higher dimensional space is that it can be computationally expensive. The mapping function 𝜙 has to be applied to each data point, and then we still have to perform our calculations on our data with the new features included. The computational costs can grow exponentially when dealing with large amounts of data and the addition of many new features.Fortunately for us, kernels come in to save the day. Since we only need the inner products of our data points to calculate the decision barrier for Support Vector Machines, which is a common classification model, kernels allow us to skip the process of mapping our data onto a higher dimensional space and calculate the inner product directly.<br/>

There are a few requirements functions have to fulfill in order to be considered a kernel.<br/>
The function needs to be continuous, meaning that can’t have any missing points in its domain<br/>
It has to be symmetric, meaning that K(x, y) = K(y, x)<br/>
It has positive semi-definiteness. This means that the kernel a symmetric matrix with non-negative eigenvalues.<br/>


**Mercer’s theorem** <br/>
Below from https://towardsdatascience.com/understanding-support-vector-machine-part-2-kernel-trick-mercers-theorem-e1e6848c6c4d<br/>
Apart from this predefined kernels, what conditions determine which functions can be considered as Kernels? This is given by Mercer’s theorem. <br/>
First condition is rather trivial i.e. the Kernel function must be symmetric. <br/>
in a finite input space, if the Kernel matrix (also known as Gram matrix) is positive semi-definite then, the matrix element i.e. the function K can be a kernel function<br/>

**SVMs Resistant overfitting** <br/>
Below from https://stats.stackexchange.com/questions/35276/svm-overfitting-curse-of-dimensionality <br/>
In practice, the reason that SVMs tend to be resistant to over-fitting, even in cases where the number of attributes is greater than the number of observations, is that it uses regularization. They key to avoiding over-fitting lies in careful tuning of the regularization parameter, C, and in the case of non-linear SVMs, careful choice of kernel and tuning of the kernel parameters.<br/>

The SVM is an approximate implementation of a bound on the generalization error, that depends on the margin (essentially the distance from the decision boundary to the nearest pattern from each class), but is independent of the dimensionality of the feature space (which is why using the kernel trick to map the data into a very high dimensional space isn't such a bad idea as it might seem). So in principle SVMs should be highly resistant to over-fitting, but in practice this depends on the careful choice of C and the kernel parameters. Sadly, over-fitting can also occur quite easily when tuning the hyper-parameters as well.<br/>

**1.6 Kernel Methods & SVMs Summary**<br/>
Margin: for generalization, find linear separator maximizing the margin. Finding the maximum margin with quadratic programming. Support vectors were the data points that were necessary for defining the maximum margin separator<br/>
Kernel trick: K(X,Y) needs domain knowledge. Kernels have to satisfy Mercer Condition tho it's still okay to use in practice even not meet<br/><br/>

**1.7 Boosting And Overfitting**<br/>
The reason why boosting is robust to overfitting even with adding more learners is that adding more learners increases the margin (the distance between boundaries), while maintaining constant error. Increasing the margin results in higher confidence in the prediction.<br/>
Boosting tends to overfit if the underlying weak learner uses an algorithm that tends to overfit (e.g. a Neural Network with a lot of hidden layers).<br/>


### 2. Support Vector Machines (SVMs) from StatQuest<br/>
**Bias**: the inability for a machine learning method to capture the true relationship is called bias. <br/>
**Variance**: In ML, the difference in fits between data set is called variance. Producing consistent predictions across different datasets means the model has low variance.  <br/>
For a overfitted model, it has low bias as it fits training set well but high variability because it results in vastly different sums of squares for different datasets. <br/>
Three commonly used methods for finding a good point between simple and complicated models are regularization, boosting and bagging. <br/>

**Part 1:** <br/>
The shortest distance between the observations and the threshold is called the **margin**. When the threshold is halfway between the two observations, the margin is as large as it can be. Moving either direction will reduce the margin. When we use the threshold that gives us the largest margin to make classification, we are using the **Maximal Margin Classifier**. However, the maximal margin classifiers are super sensentive to outliers in the training data and makes it bad. <br/> <br/>

Can we do better? <br/>
Yes, to make a threshold that is not so sensitive to outliers we must allow misclassification. Choosing a threshold that allows misclassifications is an example of the Bias/Variance Tradeoff that plagues all of machine learning. in other words, before we allowed misclassifications, we picked a threshold that was very sensitive to the training data(low bias) and it performed poorly when we got new data (high variance). In contrast, when we picked a threshold that was less sensitive to the training data and allowed misclassification(high bias),it performs better when with new data (low variance). When we allow misclassifications, the distance between the observations and the threshold is called a **Soft Margin**. To pick up the soft margin, we use cross validation to determine how many misclassifications and observations to allow inside of the Soft Margin to get the best classification. When we uses a Soft Margin to determine the location of a threshold, then we are using **Soft Margin Classifier, aka Support Vector Classifier** to classify observations. The name Support Vector Classifier (SVC) comes from the fact that the observations on the edge and within the Soft Margin are called Support Vectors. When the data is 2-dimensional, a Support Vector Classifier is a line. When the data is 3-dimensional, the SVC forms a plane. When the data are in 4+ dimensions, the SVC is a hyperplane (flat affine subspace. All flat affine subspaces are called hyperplanes. so point/line/plane are all flat affine hyperplanes technically, but usually used above 4D). <br/> <br/>

SVC can handle outliers, and because they allow miscliassifications, they can handle overlapping classifications. However, it won't perform well when one type are in sides while another type is in the middle. Because this training dataset had so much overlap, we were unable to find a satisfying SVC to separate them. Since Maximal Margin Classifiers and Support Vector Classifiers cannot handle this type of data, we need Support Vector Machines. <br/> <br/>


**The main ideas of SVM are**: <br/>
1) Start with data in a relatively low dimension <br/>
2) Move the data into a higher dimension <br/>
3) Find Support Vector Classifier that separates the higher dimensional data into two groups <br/>

How do we decide how to transform data? <br/>
In order to make the mathematics possible, SVM use **Kernel Functions** to systematically find SVC in higher dimensions. <br/>

In the dosage example, **Polynomial Kernel** is used, which has a parameter d standing for the degree of polynomial. When d=1, the Polynomial Kernel computes the relationships between each pair of observations in 1D, and these relationships are used to find SVC. When d = 2, Polynomial Kernel computes 2d relationships between each pair of observations and those relationships are used to find SVC. In summary, the Polynomial Kernel systematically increases dimensions by setting d, the degree of polynomial, and the relationships between each pair of observations are used to find SVC. The good value of d can be found with Cross Validation. <br/>


Another very commonly used Kernel is Radial Kernel, also known as **Radial Basis Function (RBF) Kernel**. This kernel finds SVC in infinite dimensions. It behaves like a weighted Nearest Neighbor model. The closest observations (nearest neighbor) have a lot of influence on how we classify the new observation. <br/>


Kernel functions only calculate the relationships between every pair of points as if they are in the higher dimensions; they don't actually do the transformation. This trick, calculating the high-dimensional relationships without actually transforming the data to the higher dimension, is called The Kernel Trick. The kernel trick reduces the amount of computation required for SVM by avoiding the math that transforms the data from low to high dimensions, and it makes calculating relationships in the infinite dimensions used by Radial Kernel possible <br/>


In summary, when we have two categories, but no obvious linear classifier that separates them in a nice way, Support Vector mahines work by moving the data into a relatively high dimensional space and finding a relatively high dimensional Support Vector Classifier that can effectively classify the observations. <br/> <br/>


**PART 2: The Polynomial Kernel** <br/>
In this example, We used a SVM with a polynomial kernel and then find a good SVC based on the high dimensional relationship.
The kernel used looks like this : (a x b+r)^d.  a and b refer to two different observation in the dataset, r determines the coefficient of the polynomial, d sets the degree of the polynomial. In this example, let's set r = 1/2 and d = 2.<br/>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVMAdd1.JPG?raw=true">
</p>

It turns out that all we need to do to calculate the high-dimensional relationships is calculate the Dot Products between each pair of points. Since  (a x b+r)^d == (a,a^2,1/2) * (b,b^2,1/2), all we need to do is plug values into the **kernel** to get the high-dimensional relationships.<br/>

<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVMAdd2.JPG?raw=true">
</p>

To review, the Polynomial Kernel  (a x b + r)^d computes relationships between pairs of observation.  and b refer to two different observation we want to calculate the high dimensional relationship for, r determines the coefficient of the polynomial, d sets the degree of the polynomial. Note, r and d are determined using Cross-Validation. Once r and d are decided, we can plug in the observations and do the math<br/>

Note, in SVC Sklearn https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html <br/> 
Kernel coefficient is gamma Parameter.Degree Parameter(Degree of the polynomial kernel function) is degree.


**PART 3: The Radial Kernel** <br/>
skip



<h1 id="8">Module: SL7  Computational Learning Theory</h1>

### 1. Computational Learning Theory<br/>
Learning Theory gets us a formal way of addressing three important questions : 1) Define learning problems 2) Show that specific algorithms work/don’t work for specific problems => upper bound 3) Show that some problems are fundamentally hard (No algorithm in a particular class will ever be able to solve them => lower bound<br/>
The tools used for analyzing learning problems are similar to those used for analyzing algorithms in computing. Resources in machine learning or computational learning theory: Time, Space, Data samples<br/>
Some of the resources useful in analyzing learning algorithms are time and space just like normal algorithms. Furthermore, learning algorithms are measured in terms of the number of samples or data they need to be effective<br/>

### 2. Inductive Learning (learning from examples )<br/>
**Factors affecting the resulting inductions**:<br/>
1) Probability of successful trainings (1 − 𝛿)  =>the accuracy to which the target concept is approximated,𝛿 is the probability of failure<br/>
2) Number of training examples (𝑚) <br/>
3) Complexity of hypothesis class (Complexity of 𝐻) => Simple hypotheses might not be sufficient to express complex concepts, whereas complex hypotheses could overfit and need much more data <br/>
4) Accuracy to which the target concept is approximated (𝜀) <br/>
5) Manner in which training examples are presented => 1) Batch learning -  entire “batches” of data at once 2) Online learning - one at a time <br/>
6) Manner in which training examples are selected => randome shuffling?<br/>

<br/><br/>
### 3. Learner Learning:<br/>
Learner choose, teacher choose, nature choose (given by nature), mean teacher choose<br/>
- The learner asks questions to the teacher: The learner selects 𝑥 and asks about 𝑐(𝑥) => the learner ask to get data<br/>
- The teacher gives examples to help learner: the teacher gives (𝑥, 𝑐(𝑥)) pairs to the learner => teacher leads the learner to something good<br/>
- Fixed distribution: 𝑥 chosen from 𝐷 by nature<br/>
- Evil distribution: Intentionally misleading<br/><br/>


<!---
THe expression used in lectures are hidden in comment in this md. just use Kudrayvtsev's note for this part since I think it is more clear.
<p align="center" width="100%">
    <img width="55%" src="https://github.com/audrey617/Notes/blob/main/ML/images/learning1.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="55%" src="https://github.com/audrey617/Notes/blob/main/ML/images/learning2.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="55%" src="https://github.com/audrey617/Notes/blob/main/ML/images/learning3.JPG?raw=true">
</p>
-->

**Learner**<br/>
X: x1x2....xk  k-bit input<br/>
H: conjunctions of literals or negation<br/>
Example: if k = 5, one input can be 00010 -> h 0 or 1<br/>

For a set of hypotheses 𝐻:<br/>
The teacher can help the learner get to the right answer using only one question.<br/>
The learner would have to ask 𝑙𝑜𝑔𝐻 questions to get to the answer on its own<br/>
<br/>
Teacher with constrained queries:<br/>
The teacher has to show what's irrelevant. This can be achieved using two positive examples.<br/>
The teacher has to show what's relevant. This can be achieved using 𝑘 negative examples, where 𝑘 is the number of variables.<br/>
The answer can be achieved in 2 + 𝑘 examples<br/>
<br/>

Learner with constrained queries:<br/>
It would be very hard (exponential time) for the learner to come up with different cases for positive and negative examples on its own.<br/>
<br/>

Learner with mistake bounds:<br/>
It's very hard to learn with constrained queries, so we change the rules: 1. Input arrives 2. Learner guesses answer 3. Wrong answer charged 4. Go to 1<br/>
Total number of mistakes will be bound by a certain number.<br/>
The algorithm is:<br/>
1) Assume that each variable can be positive and negated<br/>
2) Given input, compute output<br/>
3) If wrong, set all positive variables that were 0 to absent,  and negative variables that were 1 to absent<br/>
4) Go to 2<br/>
Using these rules, we'll never make more than 𝑘 + 1 mistakes.<br/>

### 4. PAC Learning:<br/>
**Definitions**<br/>
Computational complexity: <br/>
How much computational effort the learner needs to converge to a correct hypothesis, or to the best hypothesis in the available hypotheses space<br/>

Sample complexity: - batch<br/>
How much training examples the learner needs to create a successful hypothesis<br/>

Mistake bounds: -online <br/>
How many misclassifications can a learner make over an infinite run<br/>

Version space: <br/>
Consistent learner: Produes c(x) = h(x), where c is the true concept or true hypothesis and h is the candidate hypothesis. Learning from training set. Another way is to describe it, a consistent learner is a learner that produces a hypothesis that matches the data it had seen. It is consistent with data<br/>
Version space is the space of all the consistent hypotheses<br/>
𝑉𝑆(𝑆) = {ℎ ∈ 𝐻 | ℎ(𝑥) = 𝑐(𝑥) ∀𝑥 ∈ 𝑆}<br/>

**probably approximately correct(PAC Learning) - Error of h**<br/>
Training error: Fraction of training examples misclassified by h <br/>
True error: fraction of examples that would be misclassified on sample drawen from D. Mathatically, error_D(h) = Pr_x~D [c(x) != h(x)]. That is, Error with respect to distribution D of some hypothesis h, is the probablity that if we draw the input X from that distribution that you are going to get a mismatch between the true label and the hypothesis that we are currently evaluating. This captures the notion that it is okay to misclassify examples you will never ever see. <br/>

**probably approximately correct(PAC Learning)**<br/>
C: concept class<br/>
L: Learner<br/>
H: Hypothese space<br/>
n: |H|, size of hypothesis space<br/>
D: distribution over inputs<br/>
0<= ε <= 1/2 error goal<br/>
0<= 𝛿 <= 1/2 certainty goal. With the probability 1 - 𝛿, the algorithm has to work. And by work, we mean be able to produce a true error <= epsilon<br/>

The PAC, P is probabily 1-𝛿, A is Approximately ε, and C is correct error_D(h) = 0. We cannot force ε and 𝛿 to be zero as samples from distribution  <br/>

C is PAC-learnable by L using H iff learner L will, with probability 1-𝛿, output a hypothesis h such that error_D(h) <= ε in time and samples polynomial in 1/ε, 1/𝛿 and n.<br/>

In another words, something is PAC-learnable if you can learn to get low error with high confidence that you will have a low error in polynomial time in all parameters.<br/>
To rephrase, a concept is PAC-learnable if it can be learned to a reasonable degree of correctness within a reasonable amount of time<br/>

Is there an algorithm L such that C is PAC-learnable by L using H?<br/>
1) Keep track of VS(S,H)<br/>
2) Whever stop getting samples, uniformly pick one from VS<br/>
3) However, why the number of samples it needs isn't exponential as we have 2^k possible input in this example. If we have to see all, that would be too big. We need to have it polynomial in K, not exponential in K. How we answer it? => epsilon exhaused<br/>


**Epsilon Exhausted or Epsilon Exhausted Version Space**<br/>
VS(S) is Epsilon Exhaustion iff for all h belonging to VS(S) have low error (error_D(h) ≤ ε) <br/>
Something is epsilon exhausted, a version space is epsilon exhausted exactly in the case when everyhing that you might possible choose has an error less than epsilon. So if there is anything in there has error greater than epsilon, then it is not epsilon exhausted. <br/>


**Haussler Theorem - Bound True Error**<br/>
It bounds True Error of the number of training examples that are drawn. This gives us an upper bound that the version space is not ε-exhausted after m samples: m ≥ (1/ε) * (ln|H| + ln(1/δ)). The distribution is irrelevant.<br/>
What if true concept is not in H? Still similar function<br/>
What if infinite hypothese space? Haussler Theorem cannot handle<br/>

### 5. ADDITION (NOT IN EXAM). PAC is covered in lecture. THEN What is PCA? PCA from StatQuest<br/>
We are going through principal component analysis(PCA) one step at a time using Singular Value Decomposition (SVD).<br/>
To understand what PCA does and how it works, let's go back to the dataset that only had 2 genes<br/>
**PCA1**
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca1.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca2.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca3.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca4.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca5.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca6.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca7.JPG?raw=true">
</p>

**PCA2**
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca8.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca9.JPG?raw=true">
</p>

**Scree Plot**
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca10.JPG?raw=true">
</p>

**3D PCA**
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca11.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca12.JPG?raw=true">
</p>


**Even higher Dimension PCA**
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/ML/images/pca13.JPG?raw=true">
</p>


<h1 id="9">Module: SL8  VC Dimensions</h1>

### 1. Infinite Hypothesis<br/>
Haussler Theorem states that number of examples 𝑚 ≥ 1/𝜀(𝑙𝑛|𝐻| + 𝑙𝑛1/𝛿). This formula tells us that we are okay as long as our sample size is above 1/𝜀(𝑙𝑛|𝐻| + 𝑙𝑛1/𝛿) <br/>
The problem is, Haussler Theorem breaks with infinite hypotheses spaces.<br/>
If you have an infinite set of hypotheses, the above equation will be infinite.<br/>
Linear separators, artificial neural networks and decision trees with continuous inputs all have infinite hypotheses spaces.<br/>
Note that there is a notion of syntactic hypothesis space, which is anything that can be written as the hypothesis. However, there is also semantic hypothesis or only the meaningfully different hypothesis space<br/>
<br/>

### 2. Infinite Hypothesis to Finite meaningful hypothesis - VC Dimension<br/>
We want to differentiate between hypotheses that matter (of which there are few) from the hypotheses that don’t (of which there are infinite).<br/>

VC Dimension (Vapnik-Chervonenkis Dimension) is the largest set of inputs that the hypothesis class can **shatter** (label in all possible ways)<br/>
They can relate VC dimension of a class to the amount of data that you need to be able to learn effectively in that class. So as long as the VC dimensionality is finite, even the hypothesis class is infinite, we are able to say things about how much data we need to learn <br/>

<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/vc1.JPG?raw=true">
</p>
When we want to say yes, when there exists a set of points of a certain size, since that for all labelings, no matter how we want to label it, there are some hypothesis that work.<br/>
When we want to say no, for all points, no matter how you arrange the points, there are some labeling where there is no hypothesis that's going to work.<br/>
To show a lower bound for a VC dimension, all you need to do is find a single example of inputs for which all labeling combinations can be done. For "No" case, you have to prove that no example exists.<br/>

**In summary, Interval training**:<br/>
- To prove that a specific VC dimension is possible, you need to prove that there's some set of points you can shatter.<br/>
- To prove that a specific VC dimension is not possible, you need to prove that there's no example that can be shattered<br/>
<br/>


For any n-dimensional hyperplane hypothesis class, the VC dimension will be **n + 1**.<br/>
If our hypothesis is points inside convex polygon (On edge counts as inside), VC dimension is infinite.<br/>
<br/>

**Sample Complexity & VC Dimension**:<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/vc2.JPG?raw=true">
</p>
𝑚 ≥ 1/𝜀 * (8 * 𝑉C(𝐻) * 𝑙𝑜𝑔 (13/𝜀) + 4 * 𝑙𝑜𝑔 (2/𝛿) )   where log base is 2<br/>

VC Dimension of Finite H:<br/>
𝐻 is PAC-learnable if, and only if, VC dimension is finite<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/vc3.JPG?raw=true">
</p>


<h1 id="10">Module: SL9  Bayesian Learning</h1>
We're trying to learn the best (best means most probable/likely) hypothesis 𝐻 given data and domain knowledge <br/>
The probability of hypothesis h given input data D: 𝑃𝑟(ℎ|𝐷)<br/>
We're trying to find hypothesis h with the highest probability Pr:  𝑎𝑟𝑔𝑚𝑎𝑥_ℎ ∈ 𝐻(𝑃𝑟(ℎ|𝐷))<br/>
Check https://en.wikipedia.org/wiki/Bayesian_inference

### 1. Bayes Rule<br/>
𝑃𝑟(ℎ|𝐷) = 𝑃𝑟(𝐷|ℎ)𝑃𝑟(ℎ) / 𝑃𝑟(𝐷)<br/>
𝑃𝑟(ℎ|𝐷): The probability of a specific hypothesis given input data (Posterior probability)<br/>
𝑃𝑟(𝐷|ℎ): The probability of data given the hypothesis. It's the (likelihood) of seeing some particular labels associated with input points, given a world where some hypothesis h is true.<br/>
𝑃𝑟(𝐷): The likelihood of the data under all hypotheses (A normalizing term)<br/>
𝑃𝑟(ℎ): The prior probability of a particular hypothesis. This value encapsulates our prior belief that one hypothesis is likely or unlikely compared to other hypotheses. This is basically the domain knowledge<br/>

Chain rule example<br/>
<p align="center" width="100%">
    <img width="60%" src="https://wikimedia.org/api/rest_v1/media/math/render/svg/1386ec6778f1816c3fa6e9de68f89cee2e938066">
</p>


### 2. Bayes Learning<br/>
**Bayesian Learning algorithm:**<br/>
For each ℎ ∈ 𝐻: Calculate 𝑃𝑟(ℎ|𝐷) = 𝑃𝑟(𝐷|ℎ)𝑃𝑟(ℎ) / 𝑃𝑟(𝐷) => Pr(h|D) ∝ Pr(D|h)Pr(h)<br/>
Output: ℎ = 𝑎𝑟𝑔𝑚𝑎𝑥_ℎ ∈ 𝐻(𝑃𝑟(ℎ|𝐷))<br/>

From there, we get : <br/>
**The maximum a posteriori hypothesis (MAP)**: h_map = argmax Pr(h|D) = argmax Pr(D|h)Pr(h)<br/>
**The maximum likelihood hypothesis (ML)** : h_ml = argmax Pr(D|h)  (Assume uniform distribution of h, equally likely)<br/>
However, this is not pratical since we need to look at each h, unless H size is small<br/>

**Bayesian Learning Without Noise:**<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/bayes1.JPG?raw=true">
</p>

Pr(D|h) = 1 or 0. This asks what's the probability I would see data with these labels in a universe where h is actually true<br/>



**Bayesian Learning With Noise:**<br/>
Assumption:<br/>
Given {〈𝑥𝑖, 𝑑𝑖〉} <br/>
𝑑𝑖 = 𝑓(𝑥𝑖) + 𝜀𝑖 => f is juse some functions, doesn't mean it is linear<br/>
𝜀𝑖 ~ 𝑁(0, 𝜎^2 ) => The error follws Gaussian distribution and it is IID (Independent and Identically Distributed)<br/>

Target: What is the maximum likelihood hypothesis?<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/bayes2.JPG?raw=true">
</p>   
If you're looking for the maximum likelihood hypothesis, you should minimize the sum of squared error<br/>
This model will not work if the data is corrupted with any sort of noise other than Gaussian noise<br/><br/><br/>


**Minimum Description Length:**<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/bayes3.JPG?raw=true">
</p>   

Information theory: The optimal code for some event 𝑤 with probability 𝑃r has a length of −log𝑃r<br/>
This means that in order to maximize the Maximum a Posteriori hypothesis, we need to minimize two terms that can be described as length:<br/>
**-𝑙𝑜𝑔 𝑃𝑟(ℎ)**: <br/>
"Size of h"<br/>
This is the length of the hypothesis, which is the number of bits needed to represent this hypothesis<br/>
**-𝑙𝑜𝑔 𝑃𝑟(𝐷|ℎ)**: <br/>
"misclassification, or error in general"
This is the length of the data given a particular hypothesis. If the hypothesis perfectly describes the data, so we don’t need any points. But if the hypothesis labels some points wrong, so we need the correct labels for these points to be able to come up with a better hypothesis. So basically this term captures the error.<br/><br/>

This h_map function actually tells us, the hypothesis with the maximum probability is the one that minimizes error and the size of your hypothesis. So you want the most simplest hypothesis that minimizes error. This is pretty much the Occum's razor.
This hypothesis is called the Minimum Description<br/><br/>

In reality, these error and simple usually has a tradeoff. A more complex hypothesis will drive down error. A simple hypothesis will have some error<br/><br/>

**Bayesian Classification:**<br/>
The question in classification is “What is the best label?” not the best hypothesis<br/>
To find the best label, we need to do a weighted vote for every single hypothesis in the hypotheses set, where the weight is the probability 𝑃𝑟(ℎ|𝐷) <br/>
Important takeaway is that the best hypothesis does not always provide the best label. However, allowing all hypotheses to (do a weighted) vote leads to Bayes's optimal classifier. This is another important result: on average, you cannot do better than a weighted vote from all of the hypotheses<br/>



<h1 id="11">Module: SL10 Bayesian Inference</h1>
If we added another variable (like, “Is it thundering?”), our table would double in size. For n Boolean variables, we have a massive 2^n-entry joint distribution. We can represent it a different, more-efficient that instead takes 2n entries via factoring.<br/>
**Conditional Independence**: 𝑃𝑟(𝑋 | 𝑌, 𝑍) = 𝑃𝑟(𝑋 | 𝑍),  𝑃𝑟(𝑋, 𝑌| 𝑍) =𝑃𝑟(𝑋 | 𝑍)𝑃𝑟(Y | 𝑍)   <br/>


**Bayes Network/Belief Networks**: A representation for probabilistic quantitates over complex spaces. It’s a graphical representation of the conditional independence relationships between all the variables in a joint distribution, with nodes corresponding to the variables and edges corresponding to the dependencie. 𝑃𝑟(𝑦1, … , 𝑦𝑛) = ∏𝑃𝑟(𝑦𝑖| 𝑃𝑎𝑟𝑒𝑛𝑡𝑠(𝑦𝑖))<br/>
In belief networks, we define the Parents of a variable to be the variable’s immediate predecessors in the network<br/>


**Sampling**: Calculating independent probabilities of variables in a distribution from the graph<br/>
Why sampling from a distribution is useful?<br/>
- Simulation of a complex process.<br/>
- Approximate inference: What might happen given some conditions?<br/>
- Facilitates visualizing the information provided by data<br/>

**Inferencing Rules**:
Marginalization: 𝑃𝑟(𝑥) = ∑𝑃𝑟(𝑥, 𝑦)<br/>
Chain Rule: 𝑃𝑟(𝑥, 𝑦) = 𝑃𝑟(𝑥 | 𝑦) 𝑃𝑟(𝑦)<br/>
Bayes Rule: ignore<br/><br/><br/>


**Naïve Bayes**:
Naïve Bayes classifiers are classifiers that represent a special case of the belief networks, but with stronger independence assumptions. For our classifier to be a Naïve Bayes classifier, we make the naïve assumption that every attribute variable is conditionally independent of every other attribute variable<br/>
For the classification variable 𝑉, we would like to find the most probable target value 𝑉𝑚𝑎𝑝, given the values for attributes (𝑎1, 𝑎2, … . , 𝑎𝑛). We can write the expression for 𝑉𝑚𝑎𝑝 and then use Bayes theorem to manipulate the expression as follows<br/>
𝑉𝑚𝑎𝑝 = 𝑎𝑟𝑔𝑚𝑎𝑥_𝑣𝑗 𝑃𝑟(𝑎1, 𝑎2, … . , 𝑎𝑛 | 𝑣𝑗)𝑃𝑟(𝑣𝑗)<br/>
where 𝑃𝑟(𝑎1, 𝑎2, … . , 𝑎𝑛 | 𝑣𝑗)= 𝑃𝑟(𝑎1 | 𝑣𝑗)𝑃𝑟(𝑎2 | 𝑣𝑗) … 𝑃𝑟(𝑎𝑛 | 𝑣𝑗)<br/>
𝑉𝑚𝑎𝑝 = 𝑎𝑟𝑔𝑚𝑎𝑥_𝑣𝑗 𝑃𝑟(𝑣𝑗)∏(𝑎𝑖| 𝑣𝑗)<br/><br/>

Why Naïve Bayes is useful?<br/>
- Inference is cheap: Each of the terms to be estimated is a one dimensional probability, which can be estimated with a smaller data set than the joint probability.<br/>
- Few parameters: The total number of terms to be estimated is the number of attributes 𝑛 multiplied by the number of distinct values that 𝑣 can take.<br/>
- We can estimate the parameters with labeled data.<br/>
- Connects inference and classification.<br/>
- Empirically successful and can handle missing attributes<br/>

Disadvantages:
1) Because of the strong conditional independence assumption placed on the attributes in the model, Naïve Bayes doesn’t model the inner relationships between attributes.Naive Bayes is called naive because it assumes that each input variable is independent.<br/>
2) There is also a flaw in the “counting” approach: notice that if a particular attribute has never been seen or associated with a particular v ∈ V . Then, the entire product is zero. This is not intuitive and is exactly why this doesn’t happen in practice: the probabilities need to be smoothed out so that one bad apple doesn’t spoil the bunch.<br/><br/>





<h1 id="12">Module: UL5 - Info Theory</h1>
Are these input feature vectors similar? => mutual information<br/>
Does this feature have any information? => entropy <br/>
High entropy <-> high information <-> high uncertainty<br/>
The unfair coin's output (HHHHHHH) is completely predictable without extra information. message size 0<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/information1.JPG?raw=true">
</p>
Information gain : I(x, Y) = h(Y) − h(Y|x)<br/>
<br/>
Kullback-Leibler Divergence/relative entropy<br/>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/Notes/blob/main/ML/images/information2.JPG?raw=true">
</p>





<h1 id="13">Module: UL1 - Randomized Optimization</h1>
