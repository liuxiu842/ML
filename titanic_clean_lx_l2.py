import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier

# load data
train_data = pd.read_csv('./train.csv')
test_data = pd.read_csv('./test.csv')
#train_data information
#show all columns
#pd.set_option('display.max_columns', None)
print('查看数据信息：列名、非空个数、类型等')
print(train_data.info())
print('-'*30)
print('查看数据摘要')
print(train_data.describe())
print('-'*30)
print('查看离散数据分布')
print(train_data.describe(include=['O']))
print('-'*30)
print('查看前5条数据')
print(train_data.head())
print('-'*30)
print('查看后5条数据')
print(train_data.tail())

#using average age to instead the nan
train_data['Age'].fillna(train_data['Age'].mean(), inplace=True)
test_data['Age'].fillna(test_data['Age'].mean(), inplace=True)

#using average price to instead the nan
train_data['Fare'].fillna(train_data['Fare'].mean(),inplace=True)
test_data['Fare'].fillna(test_data['Fare'].mean(),inplace=True)

#print Embarked
print(train_data['Embarked'].value_counts())

#using the most frequent Embark instead of the nan
train_data['Embarked'].fillna('S',inplace=True)
test_data['Embarked'].fillna('S',inplace=True)

#select the feature
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
train_featrues = train_data[features]
train_labels = train_data['Survived']
test_features = test_data[features]
#test_labels = test_data['Survived']
print('特征值：', train_featrues)

dvec = DictVectorizer(sparse=False)
train_featrues = dvec.fit_transform(train_featrues.to_dict(orient='record'))
print('dvec feature names:', dvec.feature_names_)

#create ID3 decision tree
clf = DecisionTreeClassifier(criterion='entropy')

#train the decision tree
clf.fit(train_featrues, train_labels)
test_features = dvec.transform(test_features.to_dict(orient='record'))

#decision tree predict
pred_labels = clf.predict(test_features)

#decision tree decision rate(base on train data)
acc_decision_tree = round(clf.score(train_featrues, train_labels), 6)
print(u'DT score准确率为 %.4lf' % acc_decision_tree)

#use K verify 
print(u'DT cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(clf, train_featrues, train_labels, cv=10)))

#-------------------------------------------------------------------------------------------------------#
print('-'*200)
#using LR Classifier
lr = LogisticRegression(solver='liblinear', multi_class='auto')
lr.fit(train_featrues, train_labels)
predict_l = lr.predict(test_features)
#predict efficient
#print(predict_l)
acc_lr = round(lr.score(train_featrues, train_labels), 6)
print(u'lr score准确率为 %.4lf' % acc_lr)
print(u'lr cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(lr, train_featrues, train_labels, cv=10)))

#-------------------------------------------------------------------------------------------------------#
print('-'*200)
#using LDA Classifier
model = LinearDiscriminantAnalysis()
model.fit(train_featrues, train_labels)
predict_l = model.predict(test_features)
#predict efficient
#print(predict_l)
acc_lda = round(model.score(train_featrues, train_labels), 6)
print(u'LDA score准确率为 %.4lf' % acc_lda)
print(u'LDA cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_featrues, train_labels, cv=10)))

#-------------------------------------------------------------------------------------------------------#
print('-'*200)
#using GaussianNB Classifier
model = GaussianNB()
model.fit(train_featrues, train_labels)
predict_l = model.predict(test_features)
#predict efficient
#print(predict_l)
acc_nvi = round(model.score(train_featrues, train_labels), 6)
print(u'GaussianNB score准确率为 %.4lf' % acc_nvi)
print(u'GaussianNB cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_featrues, train_labels, cv=10)))

#-------------------------------------------------------------------------------------------------------#
print('-'*200)
#using SVM Classifier
model = svm.SVC(kernel='rbf', C=1.0, gamma='auto')
model.fit(train_featrues, train_labels)
predict_l = model.predict(test_features)
#predict efficient
#print(predict_l)
acc_svm = round(model.score(train_featrues, train_labels), 6)
print(u'SVM score准确率为 %.4lf' % acc_svm)
print(u'SVM cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_featrues, train_labels, cv=10)))


#-------------------------------------------------------------------------------------------------------#
print('-'*200)
#using KNN Classifier
model = KNeighborsClassifier()
model.fit(train_featrues, train_labels)
predict_l = model.predict(test_features)
#predict efficient
#print(predict_l)
acc_knn = round(model.score(train_featrues, train_labels), 6)
print(u'KNN score准确率为 %.4lf' % acc_knn)
print(u'KNN cross_val_score准确率为 %.4lf' % np.mean(cross_val_score(model, train_featrues, train_labels, cv=10)))