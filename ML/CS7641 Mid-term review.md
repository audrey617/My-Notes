# Lesson outline
- [Module: ML is the ROX](#1)
- [Module: SL1 Decision Tree](#2)
- [Module: SL2 Regression & Classification](#3)
- []()
- []()
- []()
- []()
- []()
- []()


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
Best selection is based on **largest Information gain or smallest entropy**<br />
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
Continuous Attributes: split attribute range into equal intervals<br/>
Repeat attribute along a path in a tree: NO for discrete attributes But YES for continuous attributes since we can ask different question on the same attribute. eg, ask age attribute "is it above 30", then ask "is it above 15" makes sense.<br/> 
When to stop: 1) Everything is cliassified correctly 2) No more attributes 3) overfitting-> cross-validation, validation curve& learning curve, pre-pruning or post-pruning  (Don't violate Occam's razor: entities should not be multiplied beyond necessity)<br/> 
Regression Tree: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html<br/> 
What to do for splitting: Need continuous outputs. Information gain is not available since it is unable to measure information on continuous values well and won't generalize well. Measure errors/mixedup things can use variance. Gain ratio is also one option <br/> 
What to do for leaves:  Average, local linear fit.<br/> 


<h1 id="3">Module: SL2 Regression & Classification</h1>
Regression: falling back to mean<br/> 
https://en.wikipedia.org/wiki/Regression_analysis

### 1. Errors
Our goal is to find the values of θ that minimize the above sum of squared errors. One of the common approach is to use calculus. This is where the gradient descent algorithm comes in handy. Also notice, how easy it is to take a derivative of this error function. So take a good look at the gradient descent algorithm document and come back here to find the linear equation that fits our data.<br/>

### 2. Polynomial Regression
General linear model. Detail see wiki link. Get weight or beta hat <br/> 

$$ W = (X^TX)^{-1}X^TY $$  

