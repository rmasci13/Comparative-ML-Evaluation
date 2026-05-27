import numpy as np
from utils import create_learning_curve
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt




#######################
###### NOT USED #######
#######################






def tuning(X_scaled_train, y_train, cv):
    # Hyperparameter tuning for C
    c_values = np.logspace(-1, 3, 5)
    train_scores = []
    val_scores = []

    print("Beginning SVM C Loop")
    for c in c_values:
        print(f"Loop for {c}")
        model = SVC(C=c, class_weight='balanced', random_state=42, cache_size=1000)

        # Cross validation scores
        cv_scores = cross_val_score(model, X_scaled_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_scaled_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_scaled_train), average="macro"))
    print("SVM C Loop Complete")
    # Plot
    plt.figure()
    plt.plot(c_values, train_scores, label='Training F1')
    plt.plot(c_values, val_scores, label='Validation F1')
    plt.xlabel('C values')
    plt.ylabel('Macro F1 Score')
    plt.title('SVM C Hyperparameter Tuning Curve')
    plt.xscale('log')
    plt.legend()
    plt.savefig('../plots/a_svm_c_tuning.png')
    plt.close()

    # Grid search for optimal model
    print("Beginning SVM GridSearchCV")
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': [0.001, 0.01, 0.1, 1],
        'kernel': ['rbf']
    }

    base_model = SVC(random_state=42, cache_size=1000, class_weight='balanced')

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        return_train_score=True,
        n_jobs=-1
    )
    grid_search.fit(X_scaled_train, y_train)

    print(f"Best C Value: {grid_search.best_params_['C']}")
    print(f"Best Gamma: {grid_search.best_params_['gamma']}")
    print(f"Best Score: {grid_search.best_score_}")
    print("SVM GridSearchCV Complete")

    # Hyperparameter tuning for gamma
    gamma_values = np.logspace(-3, 1, 5)
    train_scores = []
    val_scores = []

    print("Beginning SVM Gamma Loop")
    for gamma in gamma_values:
        print(f"Loop for {gamma}")
        model = SVC(gamma=gamma, C=grid_search.best_params_['C'], random_state=42, cache_size=1000)

        # Cross validation scores
        cv_scores = cross_val_score(model, X_scaled_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_scaled_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_scaled_train), average="macro"))

    # Plot
    plt.figure()
    plt.plot(gamma_values, train_scores, label='Training F1')
    plt.plot(gamma_values, val_scores, label='Validation F1')
    plt.xlabel('Gamma values')
    plt.ylabel('Macro F1 Score')
    plt.title('SVM Gamma Hyperparameter Tuning Curve')
    plt.xscale('log')
    plt.legend()
    plt.savefig('../plots/a_svm_gamma_tuning.png')
    plt.close()

    print("SVM Gamma Loop Complete")

    best_model = grid_search.best_estimator_
    train_score = f1_score(y_train, best_model.predict(X_scaled_train), average="macro")
    val_score = grid_search.best_score_

    return best_model, train_score, val_score

def svm_run(X_scaled_train, y_train, cv):
    print("Start SVM Animal")
    best_model, train_f1_score, val_f1_score = tuning(X_scaled_train, y_train, cv=cv)
    create_learning_curve(best_model, X_scaled_train, y_train, cv, 'SVM Optimal Learning Curve',
                          '../plots/a_SVM_learning_curve_optimal.png', 'f1_macro')
    print(f'Done SVM Animal - Training F1: {train_f1_score}, Validation F1: {val_f1_score}')
    return best_model, train_f1_score, val_f1_score