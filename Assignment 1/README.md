# Regression Project

**Author:** James Levi, M.S. in Artificial Intelligence
**Course:** Machine Learning – Regression and Classification
**Due Date:** September 10, 2025

---

## Overview

This project explores the application of regression and classification techniques to small datasets in order to evaluate the effects of model complexity, overfitting, and generalization. The work includes implementations of **linear regression**, **polynomial regression** of varying degrees, and **logistic regression** for two-class classification. All models were implemented from first principles using NumPy and evaluated with quantitative error metrics and visual analysis.

---

## 1. Regression Analysis

### 1.1 Linear Regression

A simple linear regression model of the form
[
d = a x + b
]
was fit to the given dataset using the **Normal Equation**:
[
w = (X^T X)^{-1} X^T d
]
where (w = [a, b]^T). The resulting parameters were used to compute the predicted outputs and the **mean squared error (MSE)**.
The linear model effectively captured the general upward trend of the data but showed moderate residual error, indicating limited flexibility for curvature.

---

### 1.2 Second-Order Polynomial Regression

To capture potential nonlinearity, the regression model was expanded to a second-order polynomial of the form:
[
d = a x^2 + b x + c
]
The inclusion of the quadratic term significantly reduced the MSE and provided a visibly improved fit compared to the linear model. The curve aligned well with the observed data without excessive oscillation.

---

### 1.3 Sixth-Order Polynomial Regression

A sixth-degree polynomial was fit to the same dataset to examine the impact of increased model complexity:
[
d = a_6 x^6 + b_6 x^5 + c_6 x^4 + d_6 x^3 + e_6 x^2 + f_6 x + g_6
]
This model achieved an extremely low MSE, nearly interpolating all data points. However, such precision comes at the cost of generalization—this is a textbook example of **overfitting**, where the model learns noise in the training data rather than the underlying relationship.

---

### 1.4 Overfitting Demonstration

To confirm the presence of overfitting, one data point ((x = 5.4857, d = 12.7972)) was removed, and the sixth-order polynomial was re-trained using the remaining seven points.
When the excluded point was reintroduced for evaluation, the MSE increased sharply (from approximately **0.108** to over **10.96**).
This substantial rise in error underscores that the sixth-order model fits the training data too tightly and lacks robustness—clear evidence of overfitting.

---

### 1.5 Model Complexity vs. Error

The mean squared error was measured for polynomial degrees ranging from 1 through 10. The resulting **MSE-versus-degree** plot demonstrates that error decreases rapidly up to degree 2, then plateaus. Beyond degree 2, additional terms provide diminishing returns and increase the risk of instability.
The optimal trade-off between bias and variance for this dataset occurs at **degree 2**, which achieves low error without unnecessary complexity.

---

## 2. Logistic Regression Classification

### 2.1 Linear Logistic Regression

A binary classification problem was modeled using logistic regression to separate two linearly separable classes in a two-dimensional feature space.
The model takes the form:
[
y = \sigma(w_1 x_1 + w_2 x_2 + w_0)
]
where (\sigma(z) = \frac{1}{1 + e^{-z}}) is the sigmoid activation.
Weights were optimized via **gradient descent** with adaptive learning rate control.
The resulting decision boundary is a straight line that perfectly separates the two classes, confirming that linear separation is sufficient for this dataset.

---

### 2.2 Quadratic Logistic Regression

To examine the effects of feature expansion, a quadratic logistic regression model was constructed by augmenting the input space with (x_1^2), (x_2^2), and (x_1x_2) terms.
The resulting decision boundary became nonlinear but did not improve classification accuracy, as the dataset was already linearly separable.
This reinforces the principle that higher-order decision surfaces are unnecessary when a linear model provides perfect separation.

---

## 3. Results Summary

| Model Type            | Degree / Order | Mean Squared Error     | Observation                            |
| --------------------- | -------------- | ---------------------- | -------------------------------------- |
| Linear Regression     | 1              | Moderate               | Captures trend but limited flexibility |
| Polynomial Regression | 2              | Low                    | Good generalization and smooth fit     |
| Polynomial Regression | 6              | Near zero              | Overfitting observed                   |
| Logistic Regression   | Linear         | Perfect classification | Simple and sufficient                  |
| Logistic Regression   | Quadratic      | Perfect classification | Added complexity without benefit       |

---

## 4. Discussion

The results clearly demonstrate that increasing model order can improve training accuracy but often at the expense of generalization. The second-order polynomial provided the most appropriate fit, balancing precision and robustness. Similarly, in classification, the linear logistic model achieved full separation without the need for nonlinear boundaries.

This experiment highlights the importance of **bias–variance trade-off**, careful **model selection**, and understanding when additional complexity ceases to add value.

---

## 5. Implementation Details

All algorithms were implemented using **NumPy** for numerical operations and **Matplotlib** for data visualization.
The pseudoinverse and normal equation were used for regression weight computation, while logistic regression employed gradient descent optimization.
No machine learning libraries were used; all computations were written from first principles to ensure full transparency and understanding of the underlying mathematics.

---

## 6. Files Included

| File               | Description                                      |
| ------------------ | ------------------------------------------------ |
| `mini_project1.py` | Complete implementation for all sections (1a–2b) |
| `README.md`        | Project documentation and analysis summary       |

---

## 7. Conclusion

This project demonstrates a full workflow of model formulation, training, evaluation, and interpretation within both regression and classification contexts.
Through experimentation and quantitative assessment, it was shown that **moderate model complexity yields optimal performance** for the given data. The second-degree polynomial and linear logistic regression models represent this balance effectively.

**Authored by:**
**James Levi**
M.S. in Artificial Intelligence
Florida Atlantic University

---
