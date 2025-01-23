from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

param_dist = {'max_depth': [3, 5, 7], 'min_samples_split': randint(2, 10)}
random_search = RandomizedSearchCV(DecisionTreeClassifier(), param_dist, n_iter=10, cv=5)
random_search.fit(X_train, y_train)
print("Best parameters:", random_search.best_params_)