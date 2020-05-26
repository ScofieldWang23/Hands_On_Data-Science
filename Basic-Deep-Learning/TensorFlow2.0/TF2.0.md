<!-- TOC -->

- [Description](#description)
- [Google Colab](#google-colab)
- [Machine Learning and Neurons](#machine-learning-and-neurons)
- [Feedforward Neural Network](#feedforward-neural-network)
  - [Forward Propagation](#forward-propagation)
  - [Geometrical picture](#geometrical-picture)
  - [Activation functions](#activation-functions)
  - [Multiclass classification](#multiclass-classification)

<!-- /TOC -->

<br>

## Description
This folder is mainly for holding notebooks and .py files for learning TensorFlow 2.0.

The content below will cover simple instruction of how to use TensorFlow 2.0 to build some very basic machine learning models and gradually advance to state of the art concepts.

We will talk talk abour major deep learning architectures, such as Deep Neural Networks, Convolutional Neural Networks (image processing), and Recurrent Neural Networks (sequence data).

**We will use TF 2.0 to finish some tasks/projects, such as:**

- Natural Language Processing (NLP)
- Recommender Systems
- Transfer Learning for Computer Vision
- Generative Adversarial Networks (GANs)
- Deep Reinforcement Learning Stock Trading Bot

**Advanced Tensorflow topics include:**

- Deploying a model with **Tensorflow Serving** (Tensorflow in the cloud)
- Deploying a model with **Tensorflow Lite** (mobile and embedded applications)
- Distributed Tensorflow training with **Distribution Strategies**
- Writing your own **customized Tensorflow model**
- Converting Tensorflow 1.x code to Tensorflow 2.0
- Constants, Variables, and Tensors
- Eager execution
- Gradient tape

**To be more specific, you’ll learn:**

- Artificial Neural Networks (ANNs) / Deep Neural Networks (DNNs)
- Convolutional Neural Networks (CNNs)
  - Computer Vision
  - Image Recognition
- Recurrent Neural Networks (RNNs)
  - NLP tasks
  - Predict Stock Returns
  - Time Series Forecasting
- How to build a Deep Reinforcement Learning Stock Trading Bot
- GANs (Generative Adversarial Networks)
- Recommender Systems
- Transfer Learning to create state-of-the-art image classifiers
- Use `Tensorflow Serving` to serve your model using a RESTful API
- Use `Tensorflow Lite` to export your model for mobile (Android, iOS) and embedded devices
- Use Tensorflow's Distribution Strategies to parallelize learning
- Low-level Tensorflow, gradient tape, and how to build your own custom models

<br>

## Google Colab
[How do I install a library permanently in Colab?](https://stackoverflow.com/questions/55253498/how-do-i-install-a-library-permanently-in-colab)

[How to upload your own data to Colab](https://colab.research.google.com/drive/1MIG0-5EZGoAElw8vQb__CfeYrkvmjfUE#scrollTo=rVEHK63cWhe2)

## Machine Learning and Neurons

[Classification notebook](https://colab.research.google.com/drive/15wpce_tlt5NLAIwjFyuoaa8CQ4-kb-BZ#scrollTo=y-llhgXY5bmX)

[Regression Notebook](https://colab.research.google.com/drive/11z_CyYByuTtDwh4auDqhO4VWY0_Jlz7w#scrollTo=4Xxa6cWIDsax)

## Feedforward Neural Network
### Forward Propagation

- A feedforward neural network is consisted of many layers of neurons
- You could regard each neuron as a simple logistic regression
- There are 2 important ways to extend a single neuron:
    ![](TF_imgs/ANN.png)

**Lines to neurons:**
![](TF_imgs/Lines_to_neurons.png)

$$\begin{aligned}
A\ line&:\ ax + b \\
A\ neuron&:\ \sigma(w^Tx + b)
\end{aligned}
$$

**Multiple neurons per layer:**  
<div  align="center">
<img src="TF_imgs/neurons.png" width = "200" height = "200" align=center/>
</div>

$$z_j = \sigma(w_j^Tx + b_j),\ for\ j=1...M$$

**Vectorize the computation:**  
- Conssider z to be a vector of size M
- Shapes:
  - z is a vector of size M(shape: (M,1))
  - x is a vector of size D(shape: (D,1))
  - W is a matrix of size DxM
  - b is a vector of size M
  - $\sigma()$ is an element-wise operation
  $$\begin{aligned}
  z_j &= \sigma(w_j^Tx + b_j),\ for\ j=1...M) \\ 
  &\Downarrow \\
  z &= \sigma(W^Tx + b)
  \end{aligned}
  $$

**Input to output for an L-layer NN:**  
![](TF_imgs/L-layer_NN.png)

- For classification:

$$\begin{aligned}
z^{(1)} &= \sigma(W^{(1)T}x + b^{(1)}) \\ 
z^{(2)} &= \sigma(W^{(2)T}x + b^{(2)}) \\
z^{(3)} &= \sigma(W^{(3)T}x + b^{(3)}) \\
... \\
p(y=1|x) &= \sigma(W^{(L)T}z^{(L-1)} + b^{(L)}
\end{aligned}
$$

- For regression:
  - Simply change the output of last layer to: $\hat{y} = W^{(L)T}z^{(L-1)} + b^{(L)}$, which is very similary to linear regression

**Another perspective**
![](TF_imgs/ANN_features.png)
<br>

### Geometrical picture
[Tensorflow Playground](https://playground.tensorflow.org/)
<br>

### Activation functions
**sigmoid: $\sigma(W^Tx) = \frac{1}{1 + exp(-W^Tx)}$** 
![](TF_imgs/sigmoid.png)

**Problems with sigmoid:**  
- Recall, we prefer data input is centered around 0 and approximately the same range
- Output of sigmoid is [0, 1], center is 0.5, hence its output can never be centered around 0, and the output is the input of next layer, which means the mean of the "input" in the latter layers can never be around 0

**tanh(Hyperbolic tangent): $\tanh(W^Tx) = \frac{exp(2W^Tx - 1)}{exp(2W^Tx - 1)}$**  
![](TF_imgs/tanh.png)
- You could prove that `tanh` is just a scaled & vertically shifted version of `sigmoid`

**Problems with sigmoid & tanh:**  
- Vanishing gradient problem:
  - This is the very common issue for deep NN, suppose $output=\sigma(...\sigma(...\sigma(...)))$
  - When we do gradient descent, we need to use chain rule to multiply the derivative of the sigmoid/tanh over and over again to update the params:
$$\frac{\partial J}{\partial W^{(1)}} = \frac{\partial J}{\partial z^{(L)}} \frac{\partial z^{(L)}}{\partial z^{(L-1)}}... \frac{\partial z^{(2)}}{\partial z^{(1)}} \frac{\partial z^{(2)}}{\partial z^{(1)}}\frac{\partial z^{(1)}}{\partial W^{(1)}}$$
![](TF_imgs/sigmoid_derivative.png)
- Why are tiny derivative is a problem?
  - Suppose the max value 0.25 is multiplied by itself 5 times: $0.25^5 \approx 0.001$
  - What if we have 0.1 multiplied by itself 5 times:$0.1^5=0.00001$
  - Result: the further back the gradient propagate to the previous layer of the network, the smaller it would become!

**ReLu**
Solution to the vanish gradient problem is: **simply don't use activation functions with vanishing gradients**

Instead, use **ReLu(Rectifier Linear Unit)**
![](TF_imgs/reLu.png)

- But wait a minute, the gradient of ReLu in the left half part(x < 0) is 0!!!, which is already vanished! This phenomenon is called "dead neuron" problem
- **HOWEVER**, in deep learning, the experiment results show that it's not a problem, it works very good!

There are some variant ReLu, which try to solve the "dead neuron" problem:  
- Leaky ReLu(LReLu)
![](TF_imgs/LReLu.png)
- Exponential Linear Unit(ELU)
![](TF_imgs/ELU.png)
- Softplus: $f(x) = log(1 + e^x)$
![](TF_imgs/softplus.png)

**Default(Recommendation):** ReLu is proven to be a default good choise by many experiments

### Multiclass classification

<br>
<br>
