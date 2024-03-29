StatQuest NN playlist: [link](https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=1)<br/>

- [Part1: Inside BlackBox](#1) (include chain rule and gradient descent)
- [Part2: Backpropagation Main Ideas & Details](#2)
- [Part3: ReLU In Action](#3)
- [Part4: Multiple Inputs and Outputs](#4)
- [Part5: ArgMax and SoftMax && The SoftMax Derivative](#5)
- [Part 6: Cross Entropy](#6)
- [Part 7: Cross Entropy Derivatives and Backpropagation](#7)
- [Part 8: Image Classification with Convolutional Neural Networks (CNNs)](#8)
- [RNN](#9)
- [Long Short-Term Memory (LSTM)](#10)
- ----------------------------------------------
- [Coding - Tensor, PyTorch](#11)


<h1 id="1">Part1: Inside BlackBox</h1>

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

<h1 id="2">Part2: Backpropagation Main Ideas & Details</h1>

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


<h1 id="3">Part3: ReLU In Action</h1>

ReLU activation function is bent and not curved. This means the derivative is not defined where the function is bent. That's a problem because Gradient Descent which we use to estimate the W and B requires a derivative for all points. However, it is not a problem because we can get around this by simply defining the derivative at the bent part to be 0 or 1. It doesn't really matter. <br/>


<h1 id="4">Part4: Multiple Inputs and Outputs</h1>
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/3.JPG?raw=true?">
</p>



<h1 id="5">Part5: ArgMax and SoftMax && The SoftMax Derivative</h1>

The broad range of values makes the Raw Output harder to interpret than it needs to be. So the raw output values are sent to either an Argmax layer or softmax layer before the final decision is made.<br/>

Argmax sets the largest value to 1 and all other to 0. Thus, when we use ArgMax, the NN prediction is simply the output with a 1 in it. The problem with ArgMax is that we cannot use it to optimize the Weights and Biases in the NN because the output values are constants 0 and 1. The derivative is 0 or undefined. <br/>
<p align="center" width="100%">
    <img width="65%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/4.JPG?raw=true?">
</p>

So that leads us to SoftMax function. When people want to use ArgMax for output, they often use SoftMax for training. Regardless of how many raw output values there are, the SoftMax output values will always be between 0 and 1. The total of softmax is 1. That measn as long as outputs are mutually exclusive, the softmax output values can be interpreted as predicted "probabilities", which depends on the W and B in the NN. W and B in turn depends on the randomly selected initial values. 
Softmax has a derivative that can be used for Backpropagation. So we can use SoftMax for training and then ArgMax to understand output. Note, when we use the SoftMax function, because the output values are between 0 and 1, we often use **Cross Entropy** instead of SSR.<br/> 
<p align="center" width="100%">
    <img width="55%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/5.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="90%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/6.JPG?raw=true?">
</p>



<h1 id="6">Part 6: Cross Entropy</h1>

When we have a simple NN with a single output, we commonly use SSR to determine how well the NN fits the data. However, when with multiple output values, we need Cross Entropy. The NN only needs a simplified form of this general Cross Entropy equation.
<p align="center" width="100%">
    <img width="90%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/7.JPG?raw=true?">
</p>

Why not SSR?<br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/8.JPG?raw=true?">
</p>


<h1 id="7">Part 7: Cross Entropy Derivatives and Backpropagation</h1>

How to use Cross Entropy with Backpropagation<br/>

Example<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/9.JPG?raw=true?">
</p>

b3 <br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/10.JPG?raw=true?">
</p>

Derivative b3<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/11.JPG?raw=true?">
</p>

Backpropagation b3<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/12.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/13.JPG?raw=true?">
</p>

<br/>

<h1 id="8">Part 8: Image Classification with Convolutional Neural Networks (CNNs)</h1>

When the image is small (6 pixels * 6 pixels), it is possible to make a normal neural network to classify it. We simply convert this 6 * 6 grid of pixels into a single column of 36 input nodes, and then connect the input nodes to a Hidden layer. Each connection has a Weight that we have to estimate with Backpropagation. So that means we need to estimate 36 weights in order to connect to the ReLU node. However, usually the first hidden layer has more than one node, and each additional node adds an additional 36 weights to be estimated.
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/23.JPG?raw=true?">
</p>

However, a larger image would cause more than 10K weights per node in the hidden layer. This normal NN doesn't scale well. Another probelm is that this NN might not perform well if the image is shifted by one pixel. Lastly, complicated images like this teddy bear tend to have correlated pixels. for example, any brownish pixel in this image tends to be close to other brown pixels, and any white pixel tends to be near other white pixels. It might be helpful if we can take advantage of the correlation that exists among each pixel. Thus, classification of large and complicated complicated images is usually done using convolutional nerual network.
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/24.JPG?raw=true?">
</p>

convolutional nerual network
<p align="center" width="100%">
    <img width="50%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/25.JPG?raw=true?">
</p>

Filter
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/26.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/27.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="60%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/28.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/29.JPG?raw=true?">
</p>

Pooling
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/30.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/31.JPG?raw=true?">
</p>

In this example, given this image, we run it through the Filter to create the Feature Map. Then we run the Feature Map through a ReLU activation function, then we select the maximum value in each area. And end up with these values in the Input Nodes. Now we multiply the values in the Input Nodes by their assoicated Weights. And add each term together and then add the bias. <br/>
**Reduce number of inputs**: starts from 6x6 image or 36 potential inputs and compressed down to 4 inputs into the NN<br/>
**Take correlations into account**: accomplished by the filter, which looks at a region of pixels, instead of just one at a time<br/>
**Can tolerate small shifts of pixels**: can still decide the same category<br/>
No matter how fancy the Convolutional neural network is, it is based on 1) filters, aka, Convolution 2) apply an activation function to the filter output 3) pooling the output of the activation function<br/>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/32.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="80%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/33.JPG?raw=true?">
</p>

<h1 id="9">RNN</h1>
**RNN**  <br/>

<h1 id="10">Long Short-Term Memory (LSTM)</h1>
**Long Short-Term Memory (LSTM)** <br/>


<h1 id="11">Coding - Tensor, PyTorch</h1>
**Tensors for Neural Networks** <br/>
From the perspective of creating a NN, tensors are ways to store the input data (Different than what tensor is in math). Unlike normal scalars, arrays, matrices, and n-dimensional matrices, Tensors were designed to take advantage of hardware acceleration. In other words, Tensors don't just hold data in various shape but also allow for all math that we have to do with the data to be done relatively quickly. Usually, Tensors and the math they do are sped up with GPUs and TPUs (Tensor processing units). One more cool thing about Tensors is that they take care of backpropagation for you with **Automatic Differentiation**, which means you can create fanciest NN and the hard part of derivatives will be taken care of by the Tensors. 

**Introduction to PyTorch** <br/>
import<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/14.JPG?raw=true?">
</p>

init<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/15.JPG?raw=true?">
</p>

Now we need to connect init to the input, activation function and output<br/>
Forward()<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/16.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/17.JPG?raw=true?">
</p>

BasicNN<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/18.JPG?raw=true?">
</p>

Example<br/>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/19.JPG?raw=true?">
</p>

PyTorch with backpropagation<br/>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/20.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/21.JPG?raw=true?">
</p>
<p align="center" width="100%">
    <img width="100%" src="https://github.com/audrey617/My-Notes/blob/main/DL/images/dlsq/22.JPG?raw=true?">
</p>
