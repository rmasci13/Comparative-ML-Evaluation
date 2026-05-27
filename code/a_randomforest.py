import numpy as np
from utils import create_learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def tuning(X_train, y_train, cv):

    # Grid search for optimal model
    print("Beginning RandomForest GridSearchCV")
    param_grid = {
        'max_depth': range(1, 21, 2),
        'n_estimators': [50, 100, 150, 200, 250, 300]
    }

    base_model = RandomForestClassifier(class_weight="balanced", random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        return_train_score=True,
        n_jobs=-1
    )
    grid_search.fit(X_train, y_train)

    print(f"Best Max_Depth Value: {grid_search.best_params_['max_depth']}")
    print(f"Best n_estimators: {grid_search.best_params_['n_estimators']}")
    print(f"Best Score: {grid_search.best_score_}")
    print("RandomForest GridSearchCV Complete")

    # Hyperparameter tuning for max_depth
    depths = range(1, 21)
    train_scores = []
    val_scores = []

    for depth in depths:
        model = RandomForestClassifier(max_depth=depth, class_weight="balanced", random_state=42, n_jobs=-1, n_estimators=grid_search.best_params_['n_estimators'])

        # Cross validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train), average="macro"))

    # Plot
    plt.figure()
    plt.plot(depths, train_scores, label='Training F1')
    plt.plot(depths, val_scores, label='Validation F1')
    plt.xlabel('Depth')
    plt.ylabel('Macro F1 Score')
    plt.title('RandomForest max_depth Tuning Curve - Animal')
    plt.legend()
    plt.savefig('../plots/a_randomforest_depth_tuning.png')
    plt.close()

    # Hyperparameter tuning for n_estimators
    n_estimators = [50, 100, 150, 200, 250, 300]
    train_scores = []
    val_scores = []

    for n in n_estimators:
        model = RandomForestClassifier(n_estimators=n, class_weight="balanced", random_state=42, max_depth=grid_search.best_params_['max_depth'])

        # Cross validation scores
        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_train), average="macro"))

    # Plot
    plt.figure()
    plt.plot(n_estimators, train_scores, label='Training F1')
    plt.plot(n_estimators, val_scores, label='Validation F1')
    plt.xlabel('n_estimators')
    plt.ylabel('Macro F1 Score')
    plt.title('RandomForest n_estimators Tuning Curve - Animal')
    plt.legend()
    plt.savefig('../plots/a_randomforest_n_estimators_tuning.png')
    plt.close()


    best_model = grid_search.best_estimator_
    train_score = f1_score(y_train, best_model.predict(X_train), average="macro")
    val_score = grid_search.best_score_

    return best_model, train_score, val_score

def rf_run(X_train, y_train, cv):
    print("Start RandomForest Animal")
    best_model, train_f1_score, val_f1_score = tuning(X_train, y_train, cv=cv)
    create_learning_curve(best_model, X_train, y_train, cv, 'RF Optimal Model Learning Curve - Animal',
                          '../plots/a_randomforest_learning_curve_optimal.png', 'f1_macro')
    print(f'Done RandomForest Animal - Training F1: {train_f1_score}, Validation F1: {val_f1_score}')
    return best_model, train_f1_score, val_f1_score