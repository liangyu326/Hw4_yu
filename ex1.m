%Train the logistic regression model with n=200, 500 800, 1000, 1500, 2000;
a = [200,500,800,1000,1500,2000];

for i=1:6
    weights(:,i) = train(a(i),data,labels);
    
    predi(:,i) = logistic_test(data_test, weights(:,i));
    
    error(i) = norm(predi(:,i)-labels_test,1);
end