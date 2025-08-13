import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np

datafile = '/Users/guy/Desktop/Sherbrooke_Lab_Data/Raman Data/ML DATA No Outliers.csv'
data = np.genfromtxt(datafile,delimiter=',',unpack=False, skip_header = 1, usecols=(range(1,17)),dtype = float)
print(data[:,0])

X = data[:,1:]
Y = data[:,0]

le = LabelEncoder() #changing the temperature (target) values into labels rather than numbers
Y_encoded = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y_encoded, test_size=0.2)

# These are the parameter ranges we want to test.
param_grid = {
    'learning_rate':    [0.03, 0.05, 0.07], # How quickly the model learns; smaller values are slower but more robust. Too small leads to overfitting.
    'max_depth':        [6, 7, 8, 9],       # How deep each tree can grow.
    'min_child_weight': [2, 3, 4],          # Minimum number of samples required in a leaf (end of tree) node. If a leaf node the algorithm was going to make does not have this many samples, it will not make the node, and the branch that would have grown to the leaf node becomes a leaf node instead.
    'n_estimators':     [250, 300, 350, 400], # The total number of trees to build in the model.
    'subsample':        [0.7, 0.8, 0.9],    # Percentage of rows used for building each tree.
    'colsample_bytree': [0.6, 0.7, 0.8],    # Percentage of columns (features) used for building each tree.
    'gamma':            [0, 0.05, 0.1]      # A value that makes it harder for the algorithm to split a node. Weighs the increase in complexity against the improvement in performance to decide whether to split or make a leaf node.
}

#Instantiate the classifier without the parameters that are to be tuned
#The search will fill these in.
classifier = xgb.XGBClassifier(objective='multi:softmax', num_class=4, use_label_encoder=False, eval_metric='mlogloss')

# tries 25 different random combinations of the parameters in param_grid.
# uses 5-fold cross-validation to evaluate each combination. (divides dataset in 5 and trains the model on each combination of 4 parts, then testing on the fifth.)
random_search = RandomizedSearchCV(
    estimator=classifier,
    param_distributions=param_grid,
    n_iter=25,  # Number of parameter settings that are sampled.
    scoring='accuracy',
    cv=5,       # 5-fold cross-validation
    verbose=2,
    random_state=42,
    n_jobs=-1   # Use all available CPU cores
)

#Fit the random search model.
print("Starting hyperparameter search...")
random_search.fit(X_train, Y_train)
print("Search complete.")

#Print the best parameters found by the search
print("\nBest parameters found: ", random_search.best_params_)
print("Best cross-validation accuracy: {:.2f}".format(random_search.best_score_))

#Use the best estimator found by the search to make predictions
best_model = random_search.best_estimator_
y_pred = best_model.predict(X_test)

#Final evaluation on the test set
score = accuracy_score(Y_test, y_pred)
print(f"\nAccuracy on the test set with best model: {score:.4f}")