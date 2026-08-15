---
id: Linear Regression
aliases: []
tags: []
---

## Intro to Linear Regression
Simple Linear Regression is a very straightforward approach for predicting a quantitative response Y on the basis of a single predictor variable X.

$$
Y \approx \beta_0 + \beta_1 X
$$

You might read the symbol as "is approximately modeled as". We will sometimes describe by saying that we are regressing Y on X (or Y onto X).
Here \beta_0 and \beta_1 are two unknown constants that represent the intercept and slope terms in the linear model. Once we use our training data to produce estimates \hat{\beta}_0 and \hat{\beta}_1 for the model coefficients, we can predict future sales on the basis of a particular value of X by computing

$$
\hat{y} = \hat{\beta}_0 + \hat{\beta}_1
$$

where \hat{y} indicates a prediction of Y on the basis of X = x.

## Estimating the coefficients
![[Pasted image 20260701071032.png]]
Let n represent the number of observation pairs, each of which consists of a measurement of X and a measurement of Y.

#### Least square method
Prediction of Y on the basis of the i-th value of X.
The fitted value for the i-th observation is

$$
\hat{y}_i = \hat{\beta}_0 + \hat{\beta}_1 x_i
$$

The residual is defined as

$$
e_i = y_i - \hat{y}_i
$$

The residual e_i represents the i-th residual—the difference between the i-th observed value and the i-th response value predicted by our linear model.

We define the residual sum of squares (RSS) as

$$
RSS = e_1^2 + e_2^2 + \dots + e_n^2
$$

or equivalently,

$$
RSS = (y_1 - \hat{\beta}_0 - \hat{\beta}_1 x_1)^2 + (y_2 - \hat{\beta}_0 - \hat{\beta}_1 x_2)^2 + \dots + (y_n - \hat{\beta}_0 - \hat{\beta}_1 x_n)^2
$$

The least squares approach chooses \hat{\beta}_0 and \hat{\beta}_1 to minimize the RSS. Using some calculus, one can show that the minimizers are

$$
\hat{\beta}_1 = \frac{\sum_{i=1}^{n} (x_i - \overline{x})(y_i - \overline{y})}{\sum_{i=1}^{n} (x_i - \overline{x})^2}
$$

$$
\beta_0 = \overline{y} - \hat{\beta}_1 \overline{x}
$$

Here

$$
\overline{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$

$$
\overline{x} = \frac{1}{n} \sum_{i=1}^{n} x_i
$$

## Cost function: squared error cost function

$$
J(w,b) = \frac{1}{2m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

Here m is the number of training examples. Here we choose w and b (just \beta_1 and \beta_0 respectively) to minimize the J function for all values of y_i.
