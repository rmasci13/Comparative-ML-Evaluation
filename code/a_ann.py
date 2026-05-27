import numpy as np
from utils import create_learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt


def tuning(X_scaled_train, y_train, cv):
    # Grid search for optimal model
    param_grid = {
        'hidden_layer_sizes': [
            (50,50), (100,100),
            (100,50,50), (100,100,100), (50, 100, 50),
            (50, 50, 50, 50, 50, 50)],
        'alpha': np.logspace(-4, 0, 5)
    }

    base_model = MLPClassifier(max_iter=1000, solver='adam', random_state=42, early_stopping=True, activation='relu', learning_rate_init=0.002)

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1_macro',
        return_train_score=True,
        n_jobs=-1
    )
    grid_search.fit(X_scaled_train, y_train)

    print(f"Best Hidden Layer Sizes Value: {grid_search.best_params_['hidden_layer_sizes']}")
    print(f"Best Alpha: {grid_search.best_params_['alpha']}")
    print(f"Best Score: {grid_search.best_score_}")
    print("ANN GridSearchCV Complete")

    # Hyperparameter tuning for alpha
    alpha_values = np.logspace(-4, 0, 5)
    train_scores = []
    val_scores = []

    for alpha in alpha_values:
        model = MLPClassifier(alpha=alpha, max_iter=1000, solver='adam', random_state=42, early_stopping=True, activation='relu', learning_rate_init=0.002, hidden_layer_sizes=grid_search.best_params_['hidden_layer_sizes'])

        # Cross validation scores
        cv_scores = cross_val_score(model, X_scaled_train, y_train, cv=cv, scoring="f1_macro", n_jobs=-1)
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_scaled_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_scaled_train), average="macro"))

    # Plot
    plt.figure()
    plt.plot(alpha_values, train_scores, label='Training F1')
    plt.plot(alpha_values, val_scores, label='Validation F1')
    plt.xlabel('Alpha values')
    plt.ylabel('Macro F1 Score')
    plt.title('ANN Alpha Tuning Curve - Animal')
    plt.xscale('log')
    plt.legend()
    plt.savefig('../plots/a_ann_tuning_alpha.png')
    plt.close()


    best_model = grid_search.best_estimator_
    train_score = f1_score(y_train, best_model.predict(X_scaled_train), average="macro")
    val_score = grid_search.best_score_

    return best_model, train_score, val_score

def ann_run(X_scaled_train, y_train, cv):
    print("Start ANN Animal")
    best_model, train_f1_score, val_f1_score = tuning(X_scaled_train, y_train, cv=cv)
    create_learning_curve(best_model, X_scaled_train, y_train, cv, 'ANN Optimal Model Learning Curve - Animal',
                          '../plots/a_ANN_learning_curve_optimal.png', 'f1_macro')
    print(f'Done ANN Animal - Training F1: {train_f1_score}, Validation F1: {val_f1_score}')
    return best_model, train_f1_score, val_f1_score