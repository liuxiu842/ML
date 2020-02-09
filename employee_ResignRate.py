import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score

# load data
train_data = pd.read_csv('./train.csv')
training_features = train_data.drop('Attrition', axis=1)
train_data['Attrition']=train_data['Attrition'].map(lambda x:1 if x=='Yes' else 0)
training_target = train_data['Attrition']
testing_features = pd.read_csv('./test.csv')

#feature dictVectorizer
dvec=DictVectorizer(sparse=False)
training_features=dvec.fit_transform(training_features.to_dict(orient='record'))
testing_features=dvec.transform(testing_features.to_dict(orient='record'))

print('-'*100)
#decision tree calculate
clf = DecisionTreeClassifier(criterion="entropy", max_depth=6, min_samples_leaf=4, min_samples_split=10)
clf.fit(training_features, training_target)
#decision tree predict
pred_labels = clf.predict(testing_features)
print("predict of testing train_data:", pred_labels)

#decision tree accuracy
acc_decision_tree = round(clf.score(training_features, training_target), 6)
print(u'Decision Tree score准确率为 %.4lf' % acc_decision_tree)
# using K cross verify statistic accuracy
print(u'Decision Tree cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, training_features, training_target, cv=10)))

print('-'*100)
#LogisticRegression  calculate
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr.fit(training_features, training_target)
#LogisticRegression predict
pred_labels = lr.predict(testing_features)
print("predict of testing train_data:", pred_labels)

#LogisticRegression accuracy
acc_decision_tree = round(lr.score(training_features, training_target), 6)
print(u'LogisticRegression score准确率为 %.4lf' % acc_decision_tree)
# using K cross verify statistic accuracy
print(u'LogisticRegression cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(lr, training_features, training_target, cv=10)))