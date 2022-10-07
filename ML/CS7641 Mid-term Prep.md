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

**3.4 Weak Learner**: No matter what the distribution is over data, a learner will do better than chance (better than chance: the error rate Pr_𝔻(.) is always less than a half). ∀𝔻 : Pr_𝔻(.) ≤ 1/2 − ε. ε here means a very small number. Technically you are bounded away from one half. Another way to think about that is, you always get some information from learner. Ther learner is always able to learn something. Chance would be the case where your probability is 1/2 and you actually learn nothing at all.<br/>

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

**3.6 AdaBoost**: 
The better we’re doing overall, the more we should focus on individual mistakes. On each iteration, our probability distribution adjusts to make D favor incorrect answers so that our classifier H can learn them better on the next round; it weighs incorrect results more and more as the overall model performance increases. <br/> 

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/Add1.JPG?raw=true">
</p>

<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/Add2.JPG?raw=true">
</p>


### 4.  Adaboost from StatQuest
Using Decision Trees and Random Forests to explain the three main concepts behind AdaBoost.<br/> 
Comparison 1: In a Random Forest, each time you make a tree, you make a full sized tree. Some trees might be bigger than others, but there is no predetermined maximum depth. In contrast, in a Forest of Trees(Stumps) with AdaBoost, the trees are usually just a node and two leaves. A tree with just one node and two leaves is called a stump. Stump can only use one variable to make a decision, thus, Stumps are technically "weak learners". However, that's the way AdaBoost likes it.<br/> 
Comparison 2: In a Random Forest, each tree has an equal vote on the final classification. In contrast, in a Forest of Stumps made with AdaBoost, some stumps get more say in the final classification than others. Large stumps weight better than smaller stumps.<br/> 
Comparison 3: In a Random Forest, each tree is made independently of the others. In other words, it doesn't matter if one tree is made first than other. In contrast, in a Forest of Stumps made with AdaBoost, order is important. The errosrs that the first stump makes influence how the second stump is made.  The second stump influece the third and so on. <br/> 

In summary, AdaBoost 1) combines a lot of weak learner to make classficiations. The weak learner are almost always stumps. 2) Some stumps get more say in the classfication than others. 3) Each Stump is made by taking the previous stump's mistakes into account. <br/>

<br/> <br/> <br/> <br/>

<h1 id="7">Module: SL6  Kernel Methods & SVMs</h1>

### 1. Support Vector Machines (SVMs)<br/>


### Addition. Support Vector Machines (SVMs) from StatQuest<br/>
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
    <img width="100%" src="https://github.com/audrey617/Notes/blob/main/ML/images/SVMAdd2.JPG?raw=true">
</p>

To review, the Polynomial Kernel  (a x b + r)^d computes relationships between pairs of observation.  and b refer to two different observation we want to calculate the high dimensional relationship for, r determines the coefficient of the polynomial, d sets the degree of the polynomial. Note, r and d are determined using Cross-Validation. Once r and d are decided, we can plug in the observations and do the math<br/>


**PART 3: The Radial Kernel e^(** <br/>

























<h1 id="8">Module: SL7  Comp Learning Theory</h1> T
<h1 id="9">Module: SL8  VC Dimensions</h1> F
<h1 id="10">Module: SL9  Bayesian Learning</h1> S
<h1 id="11">Module: SL10 Bayesian Inference</h1> S
