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
 * DT Representation: Represent data. Decision tree is a strcuture of nodes(condition/feature), edges(feature value, eg T/F) and leaves (output label).<br />
 * DT Algorithm: Sequence of steps to get output 1) Pick best attribute 2) Ask question 3) Follow answer path 4) repeat 1-3 until find answer<br />

## 3. Decision Trees Expressiveness
For Boolean function (AND & OR), the truth table is a path to a leaf.<br />
A AND B: A->T->B->T-> True leaf, otherwise False leaf<br />
A OR B : A->F->B->F-> FALSE leaf, otherwise True leaf<br />
<br />
For Boolean function (XOR)<br />
The tree is just another representation of the full truth table. <br />
<br />
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

### 4. ID3 (Top down, greedy appraoch, returns optimal decision tree, prefer shorter tree than long tree)
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
Our goal is to find the values of θ(coefficient) that minimize the above sum of squared errors (Mean Sqaure error. MSE). One of the common approach is to use calculus. This is where the gradient descent algorithm comes in handy. Also notice, how easy it is to take a derivative of this error function. So take a good look at the gradient descent algorithm document and come back here to find the linear equation that fits our data.<br/>

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
Advantages: 1) The model perfectly remembers the training data rather than an abstract generalizing 2) Fast. No need for learning 3)simple <br/>
Disadvantages: 1) Massive storage to query 2) No generalization and overfitting: sensitive to noise 3) Can return multiple values for the same input <br/>

### 2.KNN(K-Nearest Neighbors)




<h1 id="6">Module: SL5  Ensemble B&B</h1>




<h1 id="7">Module: SL6  Kernel Methods & SVMs</h1>
<h1 id="8">Module: SL7  Comp Learning Theory</h1>
<h1 id="9">Module: SL8  VC Dimensions</h1>
<h1 id="10">Module: SL9  Bayesian Learning</h1>
<h1 id="11">Module: SL10 Bayesian Inference</h1>
