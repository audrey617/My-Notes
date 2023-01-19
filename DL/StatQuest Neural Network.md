StatQuest NN playlist: [link](https://www.youtube.com/watch?v=CqOfi41LfDw&list=PLblh5JKOoLUIxGDQs4LFFD--41Vzf-ME1&index=1)<br/>
Some notes are directly from the previous ML reviews<br/><br/><br/>

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
