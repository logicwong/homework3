import pandas as pd
import numpy as np
import sklearn.preprocessing as preprocessing
import pydotplus
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, Birch
from sklearn import tree
from sklearn.externals.six import StringIO
from sklearn.metrics import classification_report
from sklearn import linear_model
from sklearn.cross_validation import train_test_split
from data_process import *

def logic_regress(df_train, df_test):
    X_train = df_train[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]]
    X_train[['Sex', 'Embarked']] = X_train[['Sex', 'Embarked']].apply(
        lambda X: nominal_to_values(X))
    y_train = df_train['Survived']

    X_test = df_test[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]]
    X_test[['Sex', 'Embarked']] = X_test[['Sex', 'Embarked']].apply(
        lambda X: nominal_to_values(X))
    y_test = df_test['Survived']

    clf = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    clf.fit(X_train, y_train)

    predictions = clf.predict(X_test)
    # 输出预测准确性。
    print(clf.score(X_test, y_test))
    # 输出更加详细的分类性能。
    print(classification_report(predictions, y_test, target_names=['died', 'survived']))

def decision_tree(df_train, df_test):
    X_train = df_train[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]]
    X_train[['Sex', 'Embarked']] = X_train[['Sex', 'Embarked']].apply(
        lambda X: nominal_to_values(X))
    y_train = df_train['Survived']

    X_test = df_test[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]]
    X_test[['Sex', 'Embarked']] = X_test[['Sex', 'Embarked']].apply(
        lambda X: nominal_to_values(X))
    y_test = df_test['Survived']
    # 决策树
    dt = tree.DecisionTreeClassifier(max_depth=4)
    dt = dt.fit(X_train, y_train)

    # 输出预测准确性。
    print(dt.score(X_test, y_test))
    # 输出更加详细的分类性能。
    y_predict = dt.predict(X_test)
    print(classification_report(y_predict, y_test, target_names=['died', 'survived']))
    feature_name = ["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]
    target_name = ["Survived", "Died"]
    dot_data = StringIO()
    tree.export_graphviz(dt, out_file=dot_data, feature_names=feature_name,
                         class_names=target_name, filled=True, rounded=True,
                         special_characters=True)
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    graph.write_pdf("Decision_Tree.pdf")

def Kmeans_cluster(df_train):
    X_train = df_train[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]]
    X_train = X_train.apply(lambda X: preprocessing.scale(X))
    X_train['Survived'] = None
    X_train['Survived'] = df_train['Survived']*10
    clf = KMeans(n_clusters=2)
    clf.fit(X_train)
    y_predict = clf.predict(X_train)
    visualize(df_train, y_predict, 'Kmeans')

def Birch_cluster(df_train):
    X_train = df_train[["Age", "SibSp", "Parch", "Fare", "Pclass", "Sex", "Cabin", "Embarked"]]
    X_train = X_train.apply(lambda X: preprocessing.scale(X))
    X_train['Survived'] = None
    X_train['Survived'] = df_train['Survived'] * 10
    clf = Birch(n_clusters=2)
    clf.fit(X_train)
    y_predict = clf.predict(X_train)
    visualize(df_train, y_predict, 'Birch')

def visualize(df, y, name):
    color_value = lambda a: 'r' if a == 1 else 'b'
    color_ori = [color_value(d) for d in y]
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.scatter(df['Age'], df['Sex'], df['Pclass'], c=color_ori, marker='o')
    ax.set_xlabel('Age')
    ax.set_ylabel('Sex')
    ax.set_zlabel('Pclass')
    ax.set_title(name)
    plt.show()

df_train = pd.read_csv('train.csv')
df_test = pd.merge(pd.read_csv('test.csv'), pd.read_csv('gender_submission.csv'), on='PassengerId')
# 利用随机森林填充age的缺失值
rfr = train_rfr(df_train)
df_train = set_missing_ages(rfr, df_train)
df_test = set_missing_ages(rfr, df_test)
df_test.loc[df_test['Fare'].isnull(), 'Fare'] = df_test[df_test['Pclass'] == 3]['Pclass'].median()

#将标称型数据转化为数值型数据
df_train.loc[ (df_train.Cabin.notnull()), 'Cabin' ] = 1
df_train.loc[ (df_train.Cabin.isnull()), 'Cabin' ] = 0
df_train['Embarked'].fillna('ffill', inplace=True)
df_train[['Sex','Embarked']] = df_train[['Sex','Embarked']].apply(lambda X: nominal_to_values(X))

df_test.loc[ (df_test.Cabin.notnull()), 'Cabin' ] = 1
df_test.loc[ (df_test.Cabin.isnull()), 'Cabin' ] = 0
df_test['Embarked'].fillna('ffill', inplace=True)
df_test[['Sex','Embarked']] = df_test[['Sex','Embarked']].apply(lambda X: nominal_to_values(X))

logic_regress(df_train, df_test)
decision_tree(df_train, df_test)
Kmeans_cluster(df_train)
Birch_cluster(df_train)