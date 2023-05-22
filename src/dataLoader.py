from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, ParameterGrid, train_test_split

dataset_path = 'allUsers.lcl.csv'
dataset_path = "../data/tommaso_data/allUsers.lcl.csv"
df = pd.read_csv(dataset_path)
# knn Cannot work with mining data, so replace them with a 0

# df = df.replace('?', pd.NA)
# df = df.dropna(thresh=len(df) * 0.5, axis=1)
# df = df.replace(pd.NA, 0)

df = df.replace('?', 0)


def get_user_test_traingi_set(test_size_percent):
    df2 = df.drop(['User'], axis=1)
    df2 = df2.drop(['Class'], axis=1)
    labels_Users = df['User']
    X_train, X_test, y_train, y_test, = train_test_split(df2, labels_Users, test_size=test_size_percent,
                                                         random_state=42)
    return X_train, X_test, y_train, y_test


def getdf():
    return df


def get_test_training_df(df_p, test_size_percent):
    df2 = df_p.drop(['User'], axis=1)
    df2 = df2.drop(['Class'], axis=1)
    labels_Users = df['User']
    X_train, X_test, y_train, y_test, = train_test_split(df2, labels_Users, test_size=test_size_percent,
                                                         random_state=42)
    return X_train, X_test, y_train, y_test


def save_results(clf, X_test, y_test, nclf, dataf):
    dataf.loc[nclf, 'acc'] = accuracy_score(y_test, clf.predict(X_test))
    dataf.loc[nclf, 'prec'] = precision_score(y_test, clf.predict(X_test), average='weighted')
    dataf.loc[nclf, 'recall'] = recall_score(y_test, clf.predict(X_test), average='weighted')
    dataf.loc[nclf, 'test f1 score (W)'] = f1_score(y_test, clf.predict(X_test), average='macro')
    return dataf


results_df = pd.DataFrame()


def test_model_and_save_result(model, name, X_train, X_test, y_train, y_test, results_df):
    model.fit(X_train, y_train)
    results_df = save_results(model, X_test, y_test, name, results_df)
