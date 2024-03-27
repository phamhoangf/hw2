import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Define logistic loss function
def logistic_loss(X, y, beta):
    n = len(y)
    scores = np.dot(X, beta)
    loss = -np.sum(y * scores - np.log(1 + np.exp(scores))) / n
    return loss

# Define gradient of logistic loss function
def logistic_gradient(X, y, beta):
    n = len(y)
    gradient = X.T.dot((np.exp(X.dot(beta)) / (1 + np.exp(X.dot(beta))) - y)) / n
    return gradient

# Define proximal operator for group lasso penalty
def prox_group_lasso(beta_j, lambda_val, w_j, t):
    return beta_j / (1 + t * lambda_val * w_j)

# Proximal gradient descent function
def proximal_gradient_descent(X, y, lambda_val, w, t, max_iterations=1000, tol=1e-4):
    n, p = X.shape
    beta = np.zeros(p)
    losses = []

    for i in range(max_iterations):
        gradient = logistic_gradient(X, y, beta)
        beta_new = np.zeros_like(beta)
        for j in range(len(w)):
            beta_new[j] = prox_group_lasso(beta[j], lambda_val, np.sqrt(w[j]), t)
        if np.linalg.norm(beta_new - beta) < tol:
            break
        beta = beta_new
        loss = logistic_loss(X, y, beta)
        losses.append(loss)

    return beta, losses

# Load data
X_train = pd.read_csv('movies/trainRatings.txt')
y_train = pd.read_csv('movies/trainLabels.txt')
group_labels = pd.read_csv('movies/groupLabelsPerRating.txt')
lambda_val = 5
t = 1e-4

# Solve logistic group lasso problem using proximal gradient descent
beta, losses = proximal_gradient_descent(X_train, y_train, lambda_val, group_labels, t)


optimal_loss = 336.207  # f*
plt.semilogy(range(len(losses)), [loss - optimal_loss for loss in losses])
plt.xlabel('Iterations')
plt.ylabel('Objective value difference (log scale)')
plt.title('Convergence of Proximal Gradient Descent')
plt.grid(True)
plt.show()