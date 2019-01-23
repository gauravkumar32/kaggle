Solution of kaggle data https://www.kaggle.com/uciml/red-wine-quality-cortez-et-al-2009

Below are the steps of this ML program :

1) Given different quality values as input, we first label them into below 3 categories : 0 - bad 1 - average 2 - good
2) Take 75% data to train our model and rest to test it.
3) Sclae the independenet variables.
4) Apply SVM with Gaussian kernel to create our model. Note that SVM is selected as it was giving best result than KNN and Random forest.
5) we apply K-fold cross validation ( with k= 10 ) and get accuracy mean as ~85%.
6) To get the best parameters for this SVMmodel we have used grid search and below is result : C = 2 ( regularization param used to prevent overfitting ) gamma = 0.5 ( radius of gaussian kernel )
7) Used backward elimination to get the most relevant features and applied same to get final result.

ToDo : create CAP curve to analyze accuracy
