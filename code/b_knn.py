from utils import create_learning_curve
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

def tuning(X_scaled_train, y_train, cv):
    # Grid search for optimal model
    neighbors = range(1, 26, 2)
    param_grid = {
        'n_neighbors': neighbors,
        'weights': ['uniform', 'distance']
    }

    base_model = KNeighborsClassifier()

    grid_search = GridSearchCV(
        estimator=base_model,
        param_grid=param_grid,
        cv=cv,
        scoring='f1',
        return_train_score=True
    )
    grid_search.fit(X_scaled_train, y_train)

    print(f"Best N-Neighbors: {grid_search.best_params_['n_neighbors']}")
    print(f"Best Weighting: {grid_search.best_params_['weights']}")

    # Hyperparameter tuning for number of neighbors
    train_scores = []
    val_scores = []

    for k in neighbors:
        model = KNeighborsClassifier(n_neighbors=k, weights=grid_search.best_params_['weights'])

        # Cross validation scores
        cv_scores = cross_val_score(model, X_scaled_train, y_train, cv=cv, scoring="f1")
        val_scores.append(cv_scores.mean())

        # Training score
        model.fit(X_scaled_train, y_train)
        train_scores.append(f1_score(y_train, model.predict(X_scaled_train)))

    # Plot
    plt.figure()
    plt.plot(neighbors, train_scores, label='Training F1')
    plt.plot(neighbors, val_scores, label='Validation F1')
    plt.xlabel('Number of Neighbors')
    plt.ylabel('F1 Score')
    plt.title('KNN Neighbors Tuning Curve - Baseball')
    plt.legend()
    plt.savefig('../plots/b_knn_neighbors_tuning.png')
    plt.close()



    best_model = grid_search.best_estimator_
    train_score = f1_score(y_train, best_model.predict(X_scaled_train))
    val_score = grid_search.best_score_

    return best_model, train_score, val_score

def knn_run(X_scaled_train, y_train, cv):
    print("Start KNN Baseball")
    best_model, train_f1_score, val_f1_score = tuning(X_scaled_train, y_train, cv=cv)
    create_learning_curve(best_model, X_scaled_train, y_train, cv, 'KNN Optimal Model Learning Curve - Baseball',
                          '../plots/b_KNN_learning_curve_optimal.png')
    print(f'Done KNN Baseball - Training F1: {train_f1_score}, Validation F1: {val_f1_score}')
    return best_model, train_f1_score, val_f1_score