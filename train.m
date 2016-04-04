function weights = train(n,data,labels)

data_train = data(1:n,:);
labels_train = labels(1:n,1);

weights = logistic_train(data_train, labels_train);

end