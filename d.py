import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Define logistic loss function
def logistic_loss(X, y, beta):
    n = len(y)
    scores = np.dot(X, beta[1:]) + beta[0]
    loss = -np.sum(y * scores - np.log(1 + np.exp(scores))) / n
    return loss

# Define gradient of logistic loss function
def logistic_gradient(X, y, beta):
    n = len(y)
    scores = np.dot(X, beta[1:]) + beta[0]
    gradient = np.dot(X.T, (np.exp(scores) / (1 + np.exp(scores)) - y)) / n
    return gradient

# Define proximal operator for group lasso penalty
def prox_group_lasso(beta, lambd, w, t):
    prox = np.zeros_like(beta)
    for j in range(len(w)):
        group_norm = np.linalg.norm(beta[j])
        if group_norm > 0:
            prox[j] = (1 - t * lambd * w[j] / group_norm) * beta[j]
    return prox

# Load training data
X_train = np.loadtxt('movies/trainRatings.txt', delimiter=',')
y_train = np.loadtxt('movies/trainLabels.txt')
group_labels = np.loadtxt("movies/groupLabelsPerRating.txt", delimiter=',')
group_titles = np.loadtxt("movies/groupTitles.txt", dtype=str)

# Initialize parameters
n, p = X_train.shape
J = int(np.max(group_labels))
lambd = 5
t = 1e-4
beta = np.zeros(p+1)  # Include intercept term
momentum = np.zeros(p+1)
gamma = 0.9  # Momentum parameter

# Initialize empty lists to store objective values and iterations
objective_values_nesterov = []

# Perform Nesterov accelerated proximal gradient descent
max_iterations = 1000
for iteration in range(max_iterations):
    # Compute objective value
    objective = logistic_loss(X_train, y_train, beta)
    for j in range(1, J+1):
        indices = np.where(group_labels == j)[0]  
        if len(indices) > 0:
            objective += lambd * np.sqrt(np.sum(beta[indices + 1]**2))
    objective_values_nesterov.append(objective)

    # Compute gradient
    gradient = logistic_gradient(X_train, y_train, beta)

    # Update momentum
    momentum_prev = momentum.copy()
    momentum[0] = gamma * momentum_prev[0] + t * gradient[0]  # Update momentum for the intercept term
    momentum[1:] = gamma * momentum_prev[1:] + t * gradient[1:]  # Update momentum for the feature coefficients

    # Update parameters using proximal operator with momentum
    beta_prev = beta.copy()
    beta[0] -= momentum[0]  # Update intercept term
    beta[1:] -= momentum[1:]  # Update feature coefficients
    beta[1:] = prox_group_lasso(beta[1:], lambd, np.sqrt(np.bincount(group_labels.astype(int))[1:]), t)

# Define optimal objective obtained from part (c)
optimal_objective = 336.207

# Plot convergence curve for Nesterov accelerated proximal gradient descent
plt.semilogy(np.arange(max_iterations), np.array(objective_values_nesterov) - optimal_objective, label='Nesterov Accelerated Proximal Gradient Descent')
plt.xlabel('Iteration')
plt.ylabel('Objective Value - Optimal Objective Value')
plt.title('Convergence Curve')
plt.grid(True)
plt.legend()
plt.show()