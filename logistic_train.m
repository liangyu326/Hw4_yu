function [weights,i,error] = logistic_train(data, labels, epsilon, maxiter)

if ~exist('epsilon','var')
    epsilon=1e-5;
end

if ~exist('maxiter','var')
    maxiter=500;
end

[n,d] = size(data);

weights = zeros(d,1);

for i=1:maxiter

weightsold = weights;

y = sigmoid(data*weights);

R = diag(y.*(1.-y));

H = data'*R*data;

weights = weights - pinv(H)*data'*(y - labels);

error = norm(weights -weightsold,2);

if error <= epsilon
    break;
end

end
end




