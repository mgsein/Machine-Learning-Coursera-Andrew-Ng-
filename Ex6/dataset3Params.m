function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

testValues = [0.01; 0.03; 0.1; 0.3; 1; 3; 10; 30];
results = [];

for C_loop = 1:length(testValues)
  for sigma_loop = 1:length(testValues)
    
    C_test = testValues(C_loop);
    sigma_test = testValues(sigma_loop);
    
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1, x2, sigma_test));
    
    #predictions on the cross validation set
    predictions = svmPredict(model, Xval);
    #prediction error
    error = mean(double(predictions ~= yval));
    
    fprintf("C: %f\nsigma: %f\nerror: %f\n", C_test, sigma_test, error);
    
    results = [results; C_test, sigma_test, error];
  end
end

[minError, minIndex] = min(results(:,3));

C = results(minIndex,1);
sigma = results(minIndex,2);

fprintf("\n\nLeast error:\nC: %f\nsigma: %f\nerror: %f\n", C, sigma, minError);

% =========================================================================

end
