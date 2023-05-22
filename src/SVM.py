from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from dataLoader import *

results_df_user = pd.DataFrame()
X_train, X_test, y_train, y_test = get_user_test_traingi_set(0.20)

# param_grid = {'C':[1,10,100,1000],
#               'gamma':[1,0.1,0.001,0.0001],
#               'kernel':['sigmoid','rbf']}
# param_grid = {'C': [1],
#               'gamma': [1],
#               'kernel': ['rbf']}
#
# grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
# grid.fit(X_train, y_train)
# test_model_and_save_result(grid, "SVM", X_train, X_test, y_train, y_test, results_df_user)

knn = KNeighborsClassifier(n_neighbors=1)
test_model_and_save_result(knn, "KNN", X_train, X_test, y_train, y_test, results_df_user)
print(results_df_user.sort_values(by=['acc'], ascending=False))
