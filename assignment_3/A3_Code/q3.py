import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score
from q1 import data_preprocessing
from q2 import data_splits, normalize_features

# from matplotlib import pyplot as plt

# Step 1: Create hyperparameter grids for each model
# TODO fill out below dictionaries with reasonable values
param_grid_decision_tree = {
    'criterion': ['gini', 'entropy'], # default: gini
    'max_depth': [None, 5], # default: None
    'min_samples_leaf': [1, 15], # default: 1
    'max_leaf_nodes': [None, 100], # default: None
}

param_grid_random_forest = {
    'n_estimators': [100, 150], # default: 100
    'max_depth': [None, 5], # default: None
    'bootstrap': [True, False], # default: True
}

param_grid_svm = {
    'kernel': ['poly', 'rbf'], # default: rbf
    'shrinking': [True, False], # default: True
    'C': [1, 10], # default: 1
    'tol': [1e-3], # default: 1e-3, TA said to not fine-tune
    'gamma': ['scale'], # default: scale, TA said to not fine-tune
}

# Step 2: Initialize classifiers with random_state=0
decision_tree = DecisionTreeClassifier(random_state=0)
random_forest = RandomForestClassifier(random_state=0)
svm = SVC(random_state=0)

# Step 3: Create a scorer using accuracy
scorer = 'accuracy'


# Step 4: Perform grid search for each model using 10-fold StratifiedKFold cross-validation
def perform_grid_search(model, X_train, y_train, params):
    # Define the cross-validation strategy
    strat_kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=0)

    # Grid search for the model
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=params,
        scoring=scorer,
        cv=strat_kfold
    )
    # TODO fit to the data
    grid_search.fit(X_train, y_train)

    best_param =  grid_search.best_params_
    best_score = grid_search.best_score_
    print("Best parameters are:", best_param)
    print("Best score is:", best_score)

    # Return the fitted grid search objects
    return grid_search, best_param, best_score



X, y = data_preprocessing()
X_train, X_test, y_train, y_test = data_splits(X, y)
X_train_scaled, X_test_scaled = normalize_features(X_train, X_test)

# Do Grid search for Decision Tree
grid_decision_tree, best_params_decision_tree, best_score_decision_tree  = perform_grid_search(
    decision_tree,
    X_train_scaled,
    y_train,
    param_grid_decision_tree
) # TODO

# Do Grid search for Random Forest
grid_random_forest, best_params_random_forest, best_score_random_forest  = perform_grid_search(
    random_forest,
    X_train_scaled,
    y_train,
    param_grid_random_forest
) # TODO

# Do Grid search for SVM
grid_svm, best_params_svm, best_score_svm = perform_grid_search(
    svm,
    X_train_scaled,
    y_train,
    param_grid_svm
) # TODO


# Code for Report Section

# def plot_accuracy(model, X_train, y_train, hyperparameter, values, title, is_none=False):
#     accuracies = []
#     x_labels = []

#     for value in values:
#         model.set_params(**{hyperparameter: value})
#         model.fit(X_train, y_train)
#         y_pred_train = model.predict(X_train)
#         accuracy = accuracy_score(y_train, y_pred_train)
#         accuracies.append(accuracy)
#         x_labels.append(str(value) if is_none and value is None else value)

#     plt.figure()
#     plt.plot(range(len(values)), accuracies, marker='o')
#     plt.xticks(range(len(values)), x_labels)
#     plt.title(title)
#     plt.xlabel(hyperparameter)
#     plt.ylabel('Training Accuracy')
#     plt.grid(True)
#     plt.show()

#     return accuracies

# decision_tree_accuracies = plot_accuracy(
#     decision_tree,
#     X_train_scaled,
#     y_train,
#     'max_depth',
#     [None, 5, 10],
#     "Decision Tree: Effect of max_depth on Accuracy",
#     is_none=True
# )

# random_forest_accuracies = plot_accuracy(
#     random_forest,
#     X_train_scaled,
#     y_train,
#     'n_estimators',
#     [5, 10, 30],
#     "Random Forest: Effect of n_estimators on Accuracy"
# )

# svm_accuracies = plot_accuracy(
#     svm,
#     X_train_scaled,
#     y_train,
#     'kernel',
#     ['linear', 'poly', 'rbf'],
#     "SVM: Effect of kernel on Accuracy"
# )

# decision_tree = DecisionTreeClassifier(**best_params_decision_tree, random_state=0)
# decision_tree.fit(X_train_scaled, y_train)
# y_pred_dt = decision_tree.predict(X_test_scaled)
# accuracy_dt = accuracy_score(y_test, y_pred_dt)

# random_forest = RandomForestClassifier(**best_params_random_forest, random_state=0)
# random_forest.fit(X_train_scaled, y_train)
# y_pred_rf = random_forest.predict(X_test_scaled)
# accuracy_rf = accuracy_score(y_test, y_pred_rf)

# svm = SVC(**best_params_svm, random_state=0)
# svm.fit(X_train_scaled, y_train)
# y_pred_svm = svm.predict(X_test_scaled)
# accuracy_svm = accuracy_score(y_test, y_pred_svm)

# fig, ax = plt.subplots()
# models = ['Decision Tree', 'Random Forest', 'SVM']
# accuracy = [accuracy_dt, accuracy_rf, accuracy_svm]
# colors = ['cadetblue', 'skyblue', 'steelblue']
# bar = ax.bar(models, accuracy, color=colors)

# for i in range(len(models)):
#     ax.text(i, accuracy[i], round(accuracy[i], 4), ha='center', va='bottom')

# plt.ylabel('Accuracy')
# plt.title('Accuracy of Models on Test Dataset')
# plt.show()