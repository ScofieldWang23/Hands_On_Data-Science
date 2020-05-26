<!-- TOC -->

- [Basic Deep Learning](#basic-deep-learning)
	- [Train NN](#train-nn)
		- [Gradient Descent](#gradient-descent)
		- [Momentum](#momentum)
		- [Nesterov Momentum](#nesterov-momentum)
		- [Variable and Adaptive learning rate](#variable-and-adaptive-learning-rate)
			- [Variable learning rate](#variable-learning-rate)
			- [Adaptive learning rate](#adaptive-learning-rate)
			- [RMSProp](#rmsprop)
			- [Adam Optimizer](#adam-optimizer)
	- [Choosing Hyper-parameters](#choosing-hyper-parameters)
		- [Grid Search](#grid-search)
		- [Random Search](#random-search)
	- [Weight Initialization](#weight-initialization)
		- [Vanishing and Exploding Gradients](#vanishing-and-exploding-gradients)
		- [Weight Initialization](#weight-initialization-1)
		- [Local Minimum](#local-minimum)
		- [TensorFlow basis](#tensorflow-basis)
			- [Tips of improving TensorFlow famaliarity](#tips-of-improving-tensorflow-famaliarity)
	- [Modern-Regularization-Techniques](#modern-regularization-techniques)
		- [Dropout](#dropout)
			- [Dropout Intuition](#dropout-intuition)
		- [Noise Injection](#noise-injection)
		- [Summary of Dropout](#summary-of-dropout)
	- [Batch Normalization](#batch-normalization)
		- [Exponentially-smoothed averages](#exponentially-smoothed-averages)
		- [Batch Normalization Theory](#batch-normalization-theory)
		- [Noise Perspective](#noise-perspective)

<!-- /TOC -->

# Basic Deep Learning
This folder is mainly for holding notebooks and .py files for Deep Learning.

The content below is simple introduction of some key components of Deep Learning.

## Train NN

### Gradient Descent
Gradient Descent vs Stochastic GD vs Batch GD

### Momentum
**Gradient descent without momentum:**
$$\theta_t \leftarrow \theta_{t-1} - \eta g_t$$

- if $g_t$ is 0, paramater $\theta$ won't change

**Gradient descent with momentum:**

two steps:
$$v_t \leftarrow \mu v_{t-1} - \eta g_t$$
$$\theta_t \leftarrow \theta_{t-1} + v_t$$

- typical values of $\mu$ are `0.9, 0.95, 0.99...`
- using momentum will greatly speed up the training, model will converge faster

### Nesterov Momentum
**could draw on some pictures from CMU DL**
$\eta$ is learning rate

**Regular momentum:**
$$v_t \leftarrow \mu v_{t-1} - \eta \nabla J(w_{t-1})$$
$$w_t \leftarrow w_{t-1} + v_t$$

**Nesterov momentum:**
$$v_t \leftarrow \mu v_{t-1} - \eta \nabla J(w_{t-1})$$
$$w_t \leftarrow w_{t-1} + \mu v_t - \eta \nabla J(w_{t-1})$$

In general, two momentum **have very similar performance** in terms of speed up training and loss function optimization

### Variable and Adaptive learning rate

#### Variable learning rate

- Learning rate is a function of time, e.g. $\eta(t)$, it should decrease with time
	- (1) Step decay: $\eta(t) = A \times (kt+1), 0 < k < 1$
	- (2) Exponential decay: $\eta(t) = A \times exp(kt)$
	- (3) 1\t decay: $\eta(t) = \frac{A}{kt+1}$

#### Adaptive learning rate

- AdaGrad
- Dependece of cost on each parameter is not the same, in one direction gradient might be steep, in another direction gradient might be flat
- Hence, adapt the learning rate for each parameter individually, based on how its own "learning condition"

$$cache = cache + grad^2$$
$$w_t \leftarrow w_{t-1} - \eta \frac{\nabla J}{\sqrt{cache + \epsilon}}$$

- Typical values for $\epsilon$ is small, around $10^{-8}, 10^{-10}$
- Each elemeny parameter is element-wsie updated independently of others

#### RMSProp

- It has been observed AdaGrad decreases learning rate too aggresively
- Since cache is growing too fast, let's decrease it on each update:
$$cache = decay * cache + (1 - decay) * grad^2$$
- Typical vlaues for decay: 0.99, 0.999, etc
- We say the cache is "leaky"

**Note: there is some ambiguity in the RMSProp update**

- what is the initial value of cache?
- you might assume it's 0, let `decay = 0.999`, initial `cache` = $0.001g^2$
- then, your intial update(ignoring $\epsilon$) is: $\frac{\eta g}{\sqrt{0.001g^2}}$, which is quite large due to small denominator $\sqrt{0.001g^2}$
- another solution is: initialize `cache = 1` instead, then $\Delta \theta= \eta \frac{g}{\sqrt{1 + 0.001g^2}} \approx \eta g$

#### Adam Optimizer

- it's often the go-to default for modern deep learning model
- Based on the content of [Exponentially-smoothed average](#Exponentially-smoothed-averages) talked later (``), we could see RMSProp is actually estimating *the average of the squared gradient*, that's why `RMSProp` gets its name(`RMS = "Root Mean Square"`), $cache = v(t) = decay * v_{t-1} + (1 - decay) * g^2 \approx mean(g^2)$

**Recall: Expected Value**

- mean(X) = $E(X)$
- $E(X^2)$ = 2nd moment of X
- $E(X^n)$ = nth moment of X
- So, $v \approx E(g^2)$

**1st moment:**
$$m_t = \mu m_{t-1} + (1 - \mu)g_t$$
$$m \approx E(g)$$

- `m` stands for `mean`
- `Adam` makes use of the 1st and 2nd moments of g, which explains what `Adam` stands for: `Adaptive Moment Estimation`
- Although the update for m looks a lot like normal momentum mentioned before, it's by no means the so called "RMSProp with momentum"!!!
- Adam is just **a combination of these 2 things: `m(1st moment)` and `v(2nd moment)`**

**Bias Correction**

There is one problem about exponentially-smoothed average: if the initial value of `Y(0)` is 0, then Y(1) = 0.99 * Y(0) + 0.01 * X(1) = 0.01 * X(1), which is a very small value (RMSProp has the same problem). That's why we need **bias coorection**:

$$\hat{Y(t)} = \frac{\hat{Y(t)}}{1 - decacy^t}$$

**Back to `Adam`**
$$
\begin{aligned}
m_t &= \beta_1 m_{t-1} + (1 - \beta_1)g_t &\approx E(g)\\
v(t) &= \beta_2 * v_{t-1} + (1 - \beta_2) * g^2 &\approx E(g^2) \\
\hat m_t &= \frac{m_t}{1 - \beta_1^t} \\
\hat v_t &= \frac{v_t}{1 - \beta_2^t} \\
w_{t+1} &\leftarrow w_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t + \epsilon}}
\end{aligned}
$$

- $\eta$ is intial learning rate
- Typical values for $\beta: 0.9, 0.99,...$, $\epsilon: 10^{-8}, 10^{-10}$



**Summary of `Adam`:**

- `Adam` is a adpative learning rate technique for deep learning
- It combines `RMSProp`(cache mechanism) with `momentum`(keeping track of old gradients)
- Generalize as 1st and 2nd moments
- Apply bias correction
<br>
<br>

## Choosing Hyper-parameters
Until now, known hyper-parameters to choose:

- learning rate, decay rate
- momentum
- regularization
- hidden layer size
- number of hiddern layer

### Grid Search

### Random Search
Sampling Logarithmically

## Weight Initialization
### Vanishing and Exploding Gradients
- In general, the deeper the network is, the better performance model would have. 
- If we use `Sigmoid` as the activation function, we would find the max derivative of sigmoid is 0.25, and after multiplying the gradients using chain rule several times, the gradient will soon become 0, which is the so called `Gradients Vanishing`, and the model can't further learn under this circumanstance.
<br>


### Weight Initialization
- First, initializing all w to 0 or constant is a bad idea in NN. (You can try yourself based on `sigmoid` or `tanh` as the activation function)
- Initializing **randomly and small**. (divided by sqrt(D) is also recommended)
- Initializing w too large would give you steep gradient, which would cause `NaNs`

**Method 1**

`W = np.random.randn(M1, M2) * 0.01`

**Method 2**: 

```python
var = 2 / (M1 + M2)
W = np.random.randn(M1, M2) * np.sqrt(var)
```

- Note: 1/var = average of fan-in and fan-out, this is used for `tanh()`
- Xavier (Glorot) Normal Intializer

**Method 3**:

```python
var = 1 / M1
W = np.random.randn(M1, M2) * np.sqrt(var)
```

- Note: this could also be used for `tanh()`

**Method 4**:

```python
var = 2 / M1
W = np.random.randn(M1, M2) * np.sqrt(var)
```

- Note: this is for `ReLu()`
- He Normal

For bias terms, it doesn't matter we initialize to 0 or random. Remember we mostlly care about breaking symmetry so that our model could learn meaningful things.


### Local Minimum
- Why are we unlikely to be at a real minimuum?
- Suppose we have 1 million dimensions, we have 2 choices for each dimension if derivative is 0: **min or max**
- The probability of being at a real minimum in all 1 million dimension is: $0.5^{1,000,000}$!!!

### TensorFlow basis
- sessions, intializing variables...

#### Tips of improving TensorFlow famaliarity

- Use TF to build linear regression, logistic regression, ANN
- Build any model based on the general framework of using gradient descent w.r.t model parameters
	- K-Means
	- GMM (Gaussian Mixture Model)
	- HMM (Hidden Markob Model)
	- Factor Analysis
	- MF (Matrix Factorization)
	- QDA (Quadratic Discriminant Analysis without distution assumptions): $y = X^TAX + b^TX + c$

<br><br>

## Modern-Regularization-Techniques

### Dropout
**Ensemble:**

- Train a group of prediction models, then average their predictions (regression) or take the majority vote (classification)
- Usually better performance than just a single model
- How?
	- (1) Each model is trained on a subset of the data
	- (2) Each model is trained on differentb features
- **Dropout is more like (2) method**
	- Train on subset of reatures, same as dropping nodes in NN
	- But not only at the input layer, drop nodes at each layer!
	- Use p(drop) or p(keep), p(keep) = 1 - p(drop)
	- Typical values:
		- p(keep) = 0.8 for input layer
		- p(keep) = 0.5 for hidden layer
	- **We only drop nodes during training**
	- **There is only 1 NN**, which is not exactly like traditional ensemble
	- **When doing prediction**, suppose we have 1 hidden layer NN: $X \rightarrow Z \rightarrow Y$, we just multiply by the p(keep) at that layer, which would shrink the output value of that layer. (Think about L2 regularization: it encourages weights to be small, also leading to shrunken values)
		-  X_drop = p(keep|layer1) * X
		-  Z = f(X_drop.dot(W1) + b1)
		-  Z_drop = p(keep|layer2) * Z
		-  Y = softmax(Z_drop.dow(W2) + b2)
		-  In implementation, we usually use **p(keep)**, because in **training phase**, it's simple to directly generate a **mask array** consisting of 0/1 and multiply it with the input matrix to get the dropout effects; In prediction, simply multiply the input with p(keep) of each layer
		
**Code implementation: `dropout_tensorflow.py`**

#### Dropout Intuition
How does multiplying by p(keep) emulate an ensemble of NN where p(drop) = 1 - p(keep)?

Suppose we have a simple NN (maybe we can't even call it NN):

- 3 inputs(nodes), 1 output(1 node)
- all nodes(x1,x2,x3) are 1
- p(keep) = 2/3
- since we have 3 nodes, 2 possibilities for each node (on/off or kept/dropped), we have 2^3=8 possible NN configurations
- HOWEVER, the probability of each configuration is NOT 1/8, because remember: for each node, p(keep) = 2/3.
- If we calculate the Expected value of X: 
	- $E(X) = \sum_{i=1}^8 x_ip(x_i) = 0*1/27 + 1*6/27 + 2*12/27 + 3*8/27 = 2$
- How about instead, we take a shortcut and directly multiply x by p(keep), which is exacctly what we did in the dropout implementation:
	- $Output = (1*2/3) * 1 + (1*2/3) * 1 + (1*2/3) * 1 = 2$
<br>

### Noise Injection
Without noise injection

`train(X, Y)`

With noise injection: $noise \sim \mathcal N(0, \sigma^2)$, $\sigma$ is small noise variance

`train(X + noise, Y)`

**Noise injection in weights:**

Instead of adding noise to inputs:

`Y = f(X + noise; W)`

We can also add noise to weights:

`Y = f(X; W + noise)`; $noise \sim \mathcal N(0, \sigma^2)$, $\sigma$ is small noise variance

### Summary of Dropout
- 2 different (structures) models in training and testing
- Train: randomly drop nodes (use p(keep) in implementation)
- Test: multiply each layer's input by p(keep)
- HOWEVER, in Tensorflow, it uses `inverted dropout`:
	- Train: randomly drop nodes + multiply each layer's input by 1 / p(keep) 
	- Test: DO NOTHING (just a regular feedforward NN)
- It emulates ensemble
<br><br> 

## Batch Normalization
Recall: We usually normalize our data before using ML models, i.e. `X = (X - mean) / std_dev`

Now instead doing the normalization just at the beginning, we would like to do this at each layer!

### Exponentially-smoothed averages

**Problem**: Suppose you have so much data $X_1, ... X_n$, which can't fit into memory all at the same time, then how to calculate the sample mean?

**Solution**:

- Read data on the fly and delete each data point after we saw it:
$$\bar{X_N} = \frac{1}{N}\sum_{i=1}^N X_i = \frac{1}{N}\left((N - 1)\bar{X_{N-1}}+ X_N\right) = (1 - \frac1N) \bar{X_{N-1}} + \frac1N X_N$$
- Using simple symbols:
$$Y(t) = (1 - \frac1t)Y(t-1) + \frac1t X(t)$$

- As you can see, each new X(t) has less and less influence(1/t) on Y(t), which makes sense since as t grows, total number of X's we've seen has also grown
- We also decrease the influence of the previous Y by (1 - 1/t)
- Now, let's say $\alpha(t) = 1/t$

What if we make $\alpha(t) = constant, 0<\alpha<1$, then:

$$Y(t) = (1 - \alpha)Y(t-1) + \alpha X(t)$$
This is the **Exponentially-smoothed** average if we express Y(t) only in terms of X:
$$Y(t) = (1 - \alpha)^tY(0) + \alpha \sum_{\tau = 0}^{t-1}(1 - \alpha)^{\tau}X(t - \tau)$$
So, do we still get the mean? (assuming X distribution will not change over time)
$$E[Y(t)] = (1 - \alpha)E[Y(t-1)] + \alpha E[X(t)] = (1 - \alpha)E[X] + \alpha E[X] = E[X]$$

**`Exponentially-smoothed average` is also known as `Low pass filter`**, because the high frequency changes has been filtered out. As you could see, for Exponentially-smoothed average, **most recent values matter more**, so if X is not stationary (distribution changes over time), then this actually may be a better way to estimate the mean.
<br>
<br>

### Batch Normalization Theory
- Recall for many ML algorithms, we like to normalize the input data before training: $z = \frac{(X - \mu)}{\sigma}$
- With batch norm, we do normalization at every layer of the neural net. We do normalization **before** activation function
- During training, we consider a small batch of data for each gradient descent step:
$$\begin{aligned}
X_B &= next\ batch\ of\ data \\
\mu_B &= mean(X_B) \\
\sigma_B &= std(X_B) = \sqrt{var(X_B)}\\
Y_B &= (X_B - \mu_B) / \sigma_B \ (not\ exact\ correct) \\
Y_B &= \frac{(X_B - \mu_B)}{\sqrt{\sigma_B^2 + \epsilon}}
\end{aligned}
$$
- Only applies during training, since only then will we have batches (**we'll do something else for testing**)

**Naming convention:** Here, we refer to the input to batch norm as `X`, and its output as `Y` (In general, `Y` is the target in machine learning scenario)

**Counter-Intuitive Step:**

- After the above normalization step, we also change its scale and location:
$$\begin{aligned}
\hat{X_B} &= \frac{X_B - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}} \\
Y &= \gamma \hat{X_B} + \beta
\end{aligned}
$$
- Why do we "un-standardize" the data after standardizing it?
	- Standardization may not be good(we don't know), so let gradient descent figure out what's best by updateing $\gamma, \beta$
	- Suppose standardization is good, then $\gamma=1, \beta=0$,
	- if not good, $\gamma, \beta$ should be whatever minimizes our loss function

**How to do prediction on testset:**

- keep tracking of "global mean" and "global variance" during training, and subtract those from test samples
- Here, we use `exponentially-smoothed average` for this:

for each batch B:
$$\begin{aligned}
\mu &= decay * \mu + (1 - decay) * \mu_B \\
\sigma^2 &= decacy * \sigma^2 + (1 - decay) * \sigma^2
\end{aligned}
$$

Theoretically, you could just use $\mu = mean(X_{train}), \sigma^2 = var(X_{train})$, **but it may not scale**

So, during test time:

$\mu, \sigma^2$ are collected during training
$$\begin{aligned}
\hat{X}_{test} &= \frac{X_{test} - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
Y_{test} &= \gamma \hat{X}_{test} + \beta
\end{aligned}
$$

**Why doest batch normalization actually help?**

- It could accelerate deep NN training by reducing `internal covariate shift`(distribution of input features can change during triaining, so that weights will have to adjust to compensate, which will increase training time)
- It also acts as a regularizer, since inputes won't take on extreme values, neither the weights!
<br>

### Noise Perspective
How does batch norm perform regularization

- Think of it as a kind of `noise injection` (discussed earlier)
- We use batch statistics: $\mu_B, \sigma_B$, we don't know the true values $\mu_{True}, \sigma_{True}$
- One way to think of any estimate:
	- Estimate = True value + Noise:
$$\begin{aligned}
 \mu_B &= \mu_{True} + \epsilon \\
 \sigma_B &= \sigma_{True} + \epsilon
\end{aligned}
$$








