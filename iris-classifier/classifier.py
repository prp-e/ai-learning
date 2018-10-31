import tensorflow.contrib.learn as skflow 
import tensorflow.contrib.layers as future_features 
from sklearn import datasets, metrics 

iris = datasets.load_iris() 

future_columns = [future_features.real_valued_column("", dimension=1)]
#print(future_columns) 

classifier = skflow.LinearClassifier(feature_columns=future_columns, n_classes = 3) 

classifier.fit(iris.data, iris.target)
score = metrics.accuracy_score(iris.target, classifier.predict(iris.data)) 

print("Accuracy: %f" % score) 
