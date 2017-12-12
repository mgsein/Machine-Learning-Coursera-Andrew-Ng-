function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples
n = size(X, 2);
num_labels = 10;
% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

#Unregularized Cost
un_regularized_J = ( ((X * theta)-y)' * ((X * theta)-y) ) / (2*m);

#Unregularized gradient
h = X * theta;
error_vector = (h - y);
grad = (X' * error_vector) /m;

#Regularized Cost
theta(1) = 0;
regularized = (lambda/(2*m)) * sum(theta.^2);
J = un_regularized_J + regularized;

#Regularized gradient
regularized_grad = (lambda./m) .* theta;
grad = grad + regularized_grad;


% =========================================================================

grad = grad(:);

end
