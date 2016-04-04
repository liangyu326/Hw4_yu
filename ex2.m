par  = [0, 0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1];
np=length(par);
[n,d]=size(X_test);
p = zeros(n,np); 
labels = -ones(n,np);

accuracy=zeros(np,1);
for i=1:np
 
    [w(:,i),c(i)] = logistic_l1_train(X_train,y_train,par(i));
    
    p(:,i) = sigmoid(X_test*w(:,i) + c(i)*ones(n,1));
    
    
    

           
end

ind = (p >= 0.5);
labels(ind) =1;

for i=1:np
    
    accuracy(i) = norm(labels(:,i)-y_test,1);

end

plot(par,accuracy);
    
    
    
