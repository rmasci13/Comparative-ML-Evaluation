import numpy as np
from utils import create_learning_curve
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def tuning(X_scaled_train, y_train, cv):
    # Grid search for optimal model
    c_values = np.logspace(-3, 3, 20)
    param_grid = {
        'C': c_values,
        'solver': ['lbfgs', 'liblinear', 'saga']
    }

    base_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        return_train_score=True
    )
    grid_search.fit(X_scaled_train, y_train)

    print(f"Best C Value: {grid_search.best_params_['C']}")
    print(f"Best Solver: {grid_search.best_params_['solver']}")

    # Hyperparameter tuning for C
    train_scores = []
    val_scores = []

    for c in c_values:
        model = LogisticRegression(C=c, class_weight='balanced', max_iter=1000, random_state=42, solver=grid_search.best_params_['solver'])

        # Cross validation scores
        cv_scores = cross_val_score(model, X_scaled_train, y_train, cv=cv, scoring="f1")
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_scaled_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_scaled_train)))

    # Plot
    plt.figure()
    plt.plot(c_values, train_scores, label='Training F1')
    plt.plot(c_values, val_scores, label='Validation F1')
    plt.xlabel('C values')
    plt.ylabel('F1 Score')
    plt.title('Logistic Regression C Tuning Curve - Baseball')
    plt.xscale('log')
    plt.legend()
    plt.savefig('../plots/b_logreg_c_tuning.png')
    plt.close()

    best_model = grid_search.best_estimator_
    train_score = f1_score(y_train, best_model.predict(X_scaled_train))
    val_score = grid_search.best_score_

    return best_model, train_score, val_score

def logreg_run(X_scaled_train, y_train, cv):
    print("Start LogReg Baseball")
    best_model, train_f1_score, val_f1_score = tuning(X_scaled_train, y_train, cv=cv)
    create_learning_curve(best_model, X_scaled_train, y_train, cv, 'LogReg Optimal Model Learning Curve - Baseball',
                          '../plots/b_logreg_learning_curve_optimal.png')
    print(f'Done LogReg Baseball - Training F1: {train_f1_score}, Validation F1: {val_f1_score}')
    return best_model, train_f1_score, val_f1_score