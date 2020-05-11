# Linear Regression
This folder is mainly for holding notebooks and .py files for Linear Regression.

The content below is simple introduction of some key components of linear regression.

## 1D Linear Regression

$$\hat{y_i} = ax_i + b$$

$$Error = \sum_{i=1}^N (y_i - \hat{y_i})$$

$\bar{x} = \frac{1}{N}\sum_{i=1}^Nx_iy_i$

$a = \frac{\bar{x}y - \bar{x}\bar{y}}{\bar{x^2}-\bar{x}^2}$

$b = \frac{\bar{y}\bar{x^2} - \bar{x}\bar{xy}}{\bar{x^2}-\bar{x}^2}$

$R^2 = 1 - \frac{SS_{res}}{SS_{total}}$

$SS_{res} = \sum_i^n(y_i - \hat{y_i})$

$SS_{total} = \sum_i^n(y_i - \bar{y})$

- $R^2 = 1$: This means $SS_{res} = 0$, which is the perfect situation, or we could say the model overfits the data
- $R^2 = 0$: This means $SS_{res} = SS_{total}$, which indicates the performance of your model is similar(same) as simply predicting the mean of y
- $R^2 < 0$: Well, this is really bad, it means the performance of your model is even worse than simply predicting the mean of y


## Multi-Linear Regression
X is `(N,D)` matrix:

- N = number of ssamples
- D = number of features

If we take 1 row of X, it represents 1 sample:

- its shape shoudl be `(1,D)`
- it is a "feature vector"
- BUT: In linear algebra, it is convention to think of vectors as column vectors, i.e `(D,1)`

Therefore:

- 1 sample prediction: $\hat{y_i} = w^Tx_i$
- N sample prediction: $\vec{y_{N \times D}} = X_{N \times D}w_{D \times 1}$

$$Error = \sum_{i=1}^N(y_i - \hat(y_i))^2 = \sum_{i=1}^N(y_i - w^Tx_i)^2$$

$$\frac{\partial E}{\partial w_j} = \sum_{i=1}^N 2(y_i - w^Tx_i)(-\frac{\partial(w^Tx_i)}{\partial w_j}) = \sum_{i=1}^N 2(y_i - w^Tx_i)(-x_{ij})$$

**Solving for w**

Since j=1...D, this represents D equations and D unknowns

$$
\begin{split}
\frac{\partial E}{\partial w_j} &= \sum_{i=1}^N 2(y_i - w^Tx_i)(-x_{ij}) = 0 \\
\Rightarrow \sum_{i=1}^N w^Tx_ix_{ij} &= \sum_{i=1}^N y_ix_{ij} \\
\Rightarrow w^T\sum_{i=1}^N x_ix_{ij} &= \sum_{i=1}^N y_ix_{ij} \\
\Rightarrow w^T(X^TX) &= y^TX \quad {\text{(full matrix form)}} \\
\Rightarrow (X^TX)w &= X^Ty \\\\
{\text{Notice this is in the form of:}} \\
Ax &=b \\
x &= A^{-1}b \\
{\text{Apply the same rule to find w:}} \\
(X^TX)w &= X^Ty \\
w &= (X^TX)^{-1}X^Ty \\\\
Ax = b \rightarrow x = np.linalg.solve(A,b) \\
w = (X^TX)^{-1}y \rightarrow x = np.linalg.solve(X^TX),\ X^Ty)
\end{split}
$$


## Practical ML issues
### Generalization error and overfitting issue

### Categorical inputs -- One-Hot encoding

### Probabilistic Interpretation of Squareed Error
#### Maximum Likelihood(ML)
Suppose the data is from Gaussian distribubtion, pdf of any single point $x_i$ is as follows:

$$p(y_i) = \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2} \frac{(y_i - \mu)^2}{\sigma^2}}$$

Because each data point is IID, we can write the joint likelihood probability function as:

$$p(y_1,y_2,...,y_n) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2} \frac{(y_i - \mu)^2}{\sigma^2}}$$

we can write this as a "likelihood":

$$p(Y|\mu) =  \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}e^{-\frac{1}{2} \frac{(y_i - \mu)^2}{\sigma^2}}$$

We want to find $\mu$ so that this likelihood is maximized (the data we measured is likely to have come from the distribution)

How to find $\mu$?

- $\frac{\partial p}{\partial \mu} = 0$
- Or a simpler way is to solve: $\frac{\partial log(p)}{\partial \mu} = 0$

$$\ell = logp(Y|\mu) = \sum_{i=1}^N \left[-\frac12 log(2\pi\sigma^2) - \frac12 \frac{(y_i - \mu)^2}{\sigma^2}\right] \tag{1}$$

$$E = \sum_{i=1}^N (y_i - \hat{y_i}) \tag{2}$$

Maximize (1) w.r.t $\mu$ $\Longleftrightarrow $ Minimize (2)

$$
\begin{split}
y \sim \mathcal N(w^Tx,\sigma^2) \Longleftrightarrow y = w^Tx + \epsilon, \epsilon \sim \mathcal N(0,\sigma^2)
\end{split}
$$

## Regularization
Why do we need Regularization?

- Avoid overfitting, have better generalization performance

### L2 Regularization (Ridge Regression)
$$J = \sum_{i=1}^N(y_n - \hat{y_n})^2 + \lambda|w|^2$$
$$|w|^2 = w^Tw = w_1^2 + w_2^2 + ... + w_D^2$$

#### Probabilistic Perspective
- Recall, plain squared error maximizes likelihood:
$$P(data|w)$$
- Why? Because J = negative log-likelihood
- Now we have 2 terms:
$$Likelihood \rightarrow P(Y|X,w) = \prod_{n=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}exp\left\{-\frac{1}{2\sigma^2}(y_n - w^Tx_n)\right\}$$

$$Prior \rightarrow P(w) = \sqrt\frac{\lambda}{2\pi} exp\left\{-\frac{\lambda}{2}w^Tw\right\}$$

- This is called MAP - Maximum A Posteriori, meaning we maximize the posterior $p(W|data)$

$$
\begin{split}
J_{Old} &\propto lnP(Y|X,w)
\\
J_{New} &\propto lnp(Y|X,w) - lnP(w)
\\
P(w|Y,X) &= \frac{P(Y|X,w)P(w)}{P(Y|X)}
\\
P(w|Y,X) &\propto P（Y|X,w)P(w)
\end{split}
$$

**Solving for w**
$$\begin{split}
J &= (Y - Xw)^T(Y - Xw) + \lambda w^Tw
\\
J &= Y^TY - 2Y^TXw + w^TX^TXw + \lambda w^Tw
\\
\frac{\partial J}{\partial w} &= -2X^TY + 2X^TXw + 2\lambda w = 0
\\
w &= (\lambda I + X^TX)^{-1}X^TY \\
\end{split}
$$

Notice the difference between the previous plain linear regression: there is one more term $\lambda I$


### Dummy Variable Trap
- `One-Hot Encoding`: If a category has K different values, then there will be K columns in X to represent this feature
- `Alternative: K-1 encoding`: We can alternatively save 1 column of sapce by letting all 0s represent 1 category value.
	- e.g. color = {red, green, blue}
	- red = [0,1], green = [1,0], blue =[0,0]
	- Undesirable because the efftect of blue gets abosorbed into the bias term

**Dummy Variable Trap:**

Many statistics resources suggest using K-1 encoding instead of one-hot encoding, why?

- Solution requires $(X^TX)^{-1}$, which needs to be inverted
- One-hot encoding would make the matrix not invertible
- So, why did we still use one-hot encoding?

How to deal with dummy variable trap?

1. Just use K-1 encoding instead
2. Remove the column of all 1s (i.e. bias terms)
3. Use L2 regularization (i.e. $X^TX$ is singluar, but $\lambda I + X^TX$ is not!)
4. Gradient descent(most general method, heavily used in deep learning)

**Note:** 

- Inverting a singular matrix is the *matrix equivalent* of division by 0
- Adding $\lambda I$ is the *matrix equivalent* of adding a small number to the denominator

**This problem is also sometimes referred to as `multicollinearity`**

- Data is arbitrary, you can't guarantee that your data is not correlated
- Gradient desent is the most preferred general solution


### L1 Regularization (Lasso Regression)
In general, we want D << N, however we also often encouter the situation when D > N. That's when we need to introduce L1 Regularization

- Goal: Select a small number of important features for prediction
- Few weights are non-zero, most $w_j$ will be 0

$$J_{Ridge} = \sum_{n=1}^N \sum_{n=1}^N(y_n - \hat{y_n})^2 + \lambda ||w||_2$$

$$J_{Lasso} = \sum_{n=1}^N \sum_{n=1}^N(y_n - \hat{y_n})^2 + \lambda ||w||_1$$

- L2 regularization puts a prior gaussian distribution on w. Now, we also put a prior on w, so it's also a MAP estimation of w
- Laplace distribution
$$p(w) = \frac{\lambda}{2} exp(-\lambda|w|)$$

**Solving for w**
$$\begin{split}
J &= (Y - Xw)^T(Y - Xw) + \lambda|w|
\\
J &= Y^TY - 2Y^TXw + w^TX^TXw + \lambda |w|
\\
\frac{\partial J}{\partial w} &= -2X^TY + 2X^TXw + \lambda sign(w) = 0
\end{split}
$$

$$
sign(x)=
\begin{cases}
1 & x > 0 \\
0 & x = 0 \\
-1 & x < 0
\end{cases}
$$


**However, compared to L2 regularization, we can't solve for w analytically!!! Instead, we can use gradient descent**

### Comparison of L1 & L2 Regularization
- L1 encourages a sparse solution (many w equal to 0)
- L2 encourages small weights (all close to 0, but not exactly 0)
- Both help us prevent overfitting by not fitting to noise

#### ElasticNet
$$J_{Ridge} = \sum_{n=1}^N \sum_{n=1}^N(y_n - \hat{y_n})^2 + \lambda ||w||_2$$

$$J_{Lasso} = \sum_{n=1}^N \sum_{n=1}^N(y_n - \hat{y_n})^2 + \lambda ||w||_1$$

$$J_{ElasticNet} = \sum_{n=1}^N \sum_{n=1}^N(y_n - \hat{y_n})^2 + \lambda_1 ||w||_1 + \lambda_2||w||_2$$
<br>

**Why divide by $\sqrt{D}$ when intializing w**: `w = np.random.randn(D) / np.sqrt(D)`

**Poor weight initialization can lead to poor convergence of loss !**

Think of Standardization:

- To standardize: $z = \frac{x - \mu}{\sigma}$ // z has mean 0, st 1
- Inverse transform: $x = z\sigma + \mu$ // x has mean $\mu$, and sd $\sigma$

So what doese the above code do?

- This means we want *w* to have mean 0, variance 1/D
- High-level idea: loss explodes due to large weights, i.e. variance is large, so we make it smaller
- In neural network, we typically normalize input data before passing it through the model.(i.e. make all numeric features have mean 0 and var 1)

Let's say $y = w_1x_1 + w_2x_2 + ... +w_Dx_D$. Then, output variance is:
$$
\begin{split}
var() &= var(w_1)var(x_1) + var(w_2)var(x_2) + ... + var(w_D)var(x_D) \\
&= var(w_1) + var(w_2) + ... + var(w_D) \quad \text{since $var(x_i)=1$} \\
&= Dvar(w)
\end{split} 
$$

Therefore:

- If we want $var(y) = 1$, we must have $var(w) = 1/D$, i.e. $sd(w) = 1/\sqrt{D}$
- Of couurse, this is not the only way to initialize weights, but the general rule always applies: make them small. For example, you might simply multiply w by a small number like 0.01





