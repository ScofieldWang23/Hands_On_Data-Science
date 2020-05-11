# Logistic Regression
This folder is mainly for holding notebooks and .py files for Logistic Regression.

The content below is simple introduction of some key components of Logistic Regression.

$$h(x) = w_0 + w_1x_1 + w_2x_2 + ...$$
$$h(x) = w^Tx$$

Logistic regression is:
$$y = \sigma(w^Tx)$$

Logistic regression makes the assumption that the data can be separated by a line/plane/hyper-plane

**Closed-form solution to the Bayes Classifier:**

Assumptions:

- All data from two classes, and both class data are gaussian distributed, 
- they have same covaraince and different mean

Multivarate Gaussian PDF:
$$p(x) = \frac{1}{\sqrt{(2\pi)^D|\sum|}}e^{\frac{1}{2}(x-\mu)^T\sum(x-u)}$$

Bayes' Rule:

- $p(Y|X) = p(X|Y)p(Y) / p(X)$
- $p(Y=1|X) = p(X|Y=1)p(Y=1) / p(X)$
- $p(Y=0|X) = p(X|Y=0)p(Y=0) / p(X)$
	- p(X|Y) is the Gaussian dist - we calculate it over all the data that belongs to specific class Y
	- p(Y) is jyst the frequency estimate of Y, e.g. $p(Y) = \frac{count(Y=1)}{count(Y=1) + count(Y=0)}$

Put it into the logistic regression framework:
$$\begin{split}
p(y=1|x) &= \frac{p(x|y=1)p(y=1)}{p(x)} = \frac{p(x|y=1)p(y=1)}{p(x|y=1)p(y=1) + p(x|y=0)p(y=0)} \\
p(y=1|x) &= \frac{1}{1 + \frac{p(x|y=0)p(y=0)}{p(x|y=1)p(y=1)}} \\
&= \frac{1}{1 + exp(-{w^Tx+b})} 
\end{split}
$$

$$-(w^Tx+b) = ln\left(\frac{p(x|y=0)p(y=0)}{p(x|y=1)p(y=1)}\right)$$


## Logistic regression error
In linear regression, $$J=\sum_n(t_n - y_n)^2$$, we assume Gaussian distributed error, because log(Gaussian) = squared function, which is pretty much the same thing as *J*

However, for logistic regression, error can't be Gaussian distribuuted, because:

- Target is only 0 or 1
- Ouptput is just probability between 0~1

That's why we introduce **Cross-Entropy Error**:
$$J = -\sum_n[t_nlog(y_n) + (1-t_n)log(1-y_n)]$$

- t=target, y=output of logistic regression, i.e. probability
- if t==1, only first term matters; if t==0, only second term matters
- log(y) is between -inf and 0
- we wish to minimize J

**Maximum Likelihood**:
$$L = \prod_{n=1}^n y_n^{t_n}(1 - y_n)^{1-t_n}$$
if we take the log likelihood:
$$logL = \sum_n[t_nlog(y_n) + (1-t_n)log(1-y_n)]$$
which is the negative cross-entropy error

### Derivatives
X have j features and n samples
$$\begin{split}
y_n &= \sigma(a_n) = \frac{1}{1 + e^{-a_n}} \\
\frac{\partial y_n}{\partial a_n} &= \frac{-1}{(1 + e^{-a_n})^2}(e^{-a_n})(-1) \\
&= \frac{e^{-a_n}}{(1+e^{-a_n})^2} = \frac{1}{1+e^{-a_n}} \frac{e^{-a_n}}{1+e^{-a_n}} = y_n(1 - y_n) \\
\\
a_n &= w^Tx_n = w_0x_0 + w_1x_1 + ... +w_jx_j \\
\frac{\partial a_n}{\partial w} &= x_n \\ \\
{\text{Putting all together}} \\ \\
\frac{\partial J}{\partial w} &= -\sum_{n=1}^N \frac{t_n}{y_n}y_n(1 - y_n)x_n - \frac{1 - t_n}{1 - y_n}y_n(1 - y_n)x_n \\
&= \sum_{n=1}^N (y_n - t_n)x_n \\ \\
{\text{Corresponding Vectorized representation:}} \\ \\
\frac{\partial J}{\partial w} &= X^T(Y - T)
\end{split}
$$
Shape of X: (N,D), Shape of Y,T: (N,1)

How about bias term? Because all bias term is 1, so:
$$\frac{\partial J}{\partial w_0} = \sum_{n=1}^N (y_n - t_n)x_{n0} = \sum_{n=1}^N (y_n - t_n)$$

### Interpreting weights w
Recall the interpretation of the ouutput of logistic regression:
$$\begin{split}
p(y=1|x) &= \sigma(w^Tx) = \frac{1}{1 + exp(-w^Tx)} \\
p(y=0|x) &= 1 - \sigma(w^Tx) = \frac{exp(-w^Tx)}{1 + exp(-w^Tx)} \\
Odds &= \frac{p(y=1|x)}{p(y=0|x)} \\ \\
{\text{Take the log}} \\ \\
log(odds) &= log\left(\frac{p(y=1|x)}{p(y=0|x)}\right) = w^Tx
\end{split}
$$

So, just apply the same way of linear regression to interpret logistic regression in terms of `log(odds)`

## Regularization
$$
\begin{split}
J_{Ridge} &= -\sum_n[t_nlog(y_n) + (1-t_n)log(1-y_n)] + \lambda||w||_2 \\
\frac{\partial J_{Ridge}}{\partial w} &= X^T(Y - T) +  \lambda w\\
J_{Lasso} &= -\sum_n[t_nlog(y_n) + (1-t_n)log(1-y_n)] + \lambda||w||_1 \\
\frac{\partial J_{Lasso}}{\partial w} &= X^T(Y - T) +  \lambda sign(w)\\
\end{split}
$$





