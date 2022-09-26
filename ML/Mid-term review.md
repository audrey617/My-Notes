# Lesson outline
- [Module: ML is the ROX](#1)
- [Module: SL1 Decision Tree](#2)
- []()
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
Instances: inputs, a row of data, an observation from domain <br />
Concept: A function mapping inputs to outputs<br />
Target concept: The actual answer/function we try to find<br />
Hypothesis class: The set of all concepts. All functions I am willing to think about. Could be all possible functions however that would be hard for us to figure out which one is the right one given finite data.<br />
Sample (Training set): A set of instances with correct labels<br />
Candidate: A concept that you think might be the target concept<br />
Testing set: A set of instances with correct labels that was not visible to the learning algorithm<br />

## 2. Representation VS Algorithm
DT Representation: Represent data. Decision tree is a strcuture of nodes(condition/feature), edges(feature value, eg T/F) and leaves (output label).<br />
DT Algorithm: Sequence of steps to get output 1) Pick best attribute 2) Ask question 3) Follow answer path 4) repeat 1-3 until find answer<br />

## 2. Decision Trees Expressiveness


