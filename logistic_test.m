function [predi]=logistic_test(data, weights)

n=size(data,1);
predi=zeros(n,1);
p = sigmoid(data*weights);

ind = (p >= 0.5);

predi(ind) = 1;

end
