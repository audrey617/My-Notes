StatQuest NN playlist: [link](https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=1)<br/>

**Part1: Inside BlackBox** <br/>
A neural network consists of Nodes and connection between the nodes. The numbers along each connection represent parameter values (weights and biases) that were estimated when this NN was fit to the data. It starts out with unknown values that are estimated when we fit NN to a datase using Backpropagation. Usually a neural network has more than one input/output node and different layers of nodes between input and output nodes. The layers of nodes between the input and output nodes are called hidden layers. When you build a neural network, one of the first thing you do is decide how many hidden layers you want, and how many nodes go into each hidden layer. The hidden layer nodes contains activation functions/curved bent lines. The previous layer node output with the (weights * x + biases) becomes the input in the new layer node activation function. The node has the same activation function, but the weights and biases on the connection slice them, flip and stretch them into new shapes. In the end they get added with parameter adjustment, so we get a new squiggle green line for final prediction.

<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition1.JPG?raw=true">
</p>
<br/>

**Part1.1: chain rule**: skip <br/>

**Part1.2: Gradient descent**: <br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/Notes/blob/main/DL/images/dlsq/1.JPG?raw=true">
</p>

More details go to cs229 [note](https://cs229.stanford.edu/lectures-spring2022/main_notes.pdf) 1.1 LMS algorithm. page 9 - 13<br/><br/>


**Part2: Backpropagation Main Ideas & Details** <br/>
Backpropagation optimizes the Weights and Biases in the NN. Conceptually, backpropagation starts with the last parameter (here b3) and works its way backwards to estimate all other parameters. SSR (sum of squares residuals = Σ(obs_y - predict_y) for n samples). When we optimize more than one parameter, the derivatives that we have already calculated with respect ot SSR do not change <br/>
Step1: using chain rule to calculate derivatives  <br/> 
Step2: plug the derivates into Gradient Descent to optimize parameters <br/> 

Optimize all weights and biases in this NN example. activation function is softplus. we combine these derivatives with others, and use gradient descent to optimize all simultaneously. First we initialize the W from normal distribution (or others) and bias term to 0. Then we plug in samples to get the values, and we use each derivative to calculate a step size. Next we update the parameters in the NN and repeat until the predictions no longer improve or meet other criteria. <br/>

w1, b1 Derivative <br/> 
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/Notes/blob/main/DL/images/dlsq/2.JPG?raw=true">
</p>
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_1.JPG?raw=true">
</p>
w2, b2 Derivative<br/> 
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_2.JPG?raw=true">
</p>
w3, w4, b3 Derivative<br/> 
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_3.JPG?raw=true">
</p>
get values to calculate step size<br/> 
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/Notes/blob/main/ML/images/addition2_4.JPG?raw=true">
</p>

**Part3: ReLU In Action**: skip <br/>

**Part4: Multiple Inputs and Outputs**: details skip <br/>
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/3.JPG?raw=true?">
</p>

**Part5: ArgMax and SoftMax && The SoftMax Derivative**: <br/>
The broad range of values makes the Raw Output harder to interpret than it needs to be. So the raw output values are sent to either an Argmax layer or softmax layer before the final decision is made.<br/>

Argmax sets the largest value to 1 and all other to 0. Thus, when we use ArgMax, the NN prediction is simply the output with a 1 in it. The problem with ArgMax is that we cannot use it to optimize the Weights and Biases in the NN because the output values are constants 0 and 1. <br/>
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/4.JPG?raw=true?">
</p>

So that leads us to SoftMax function. When people want to use ArgMax for output, they often use SoftMax for training. Regardless of how many raw output values there are, the SoftMax output values will always be between 0 and 1. The total of softmax is 1. That measn as long as outputs are mutually exclusive, the softmax output values can be interpreted as predicted "probabilities", which depends on the W and B in the NN. W and B in turn depends on the randomly selected initial values. 
Softmax has a derivative that can be used for Backpropagation. So we can use SoftMax for training and then ArgMax to understand output. Note, previously we used SSR to determine how well the NN fit the data. However, when we use the SoftMax function, because the output values are between 0 and 1, we often use **Cross Entropy** to determine how well the NN fits the data.<br/> 
<p align="center" width="100%">
    <img width="55%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/5.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="90%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/6.JPG?raw=true?">
</p>


**Part 6: Cross Entropy**: <br/>
