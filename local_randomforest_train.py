import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer #, KNNImputer, IterativeImputer

# import optuna


##############################################
# Load the dataset
# Assuming the dataset is a CSV file in Google Drive
data = pd.read_csv('/Users/lap16109/Downloads/ddos/new-train.csv')

##############################################
# Get unique classes
unique_classes = data['Label'].unique()

# Count the number of classes
num_classes = len(unique_classes)
# Count the number of columns
num_columns = data.shape[1]

# Data distribution for the "Label" column
label_distribution = data['Label'].value_counts()

# Print the results
print(f"Number of columns: {num_columns}")
print("\nData Distribution by 'Label':")
print(label_distribution)

# Print the results
print(f"Number of unique classes: {num_classes}")
print("Classes:", unique_classes)


##############################################

# Drop rows with NaN values (or alternatively, you could fill them with a specific value)
# data.dropna(inplace=True)
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Explicitly name the target column
target_column = 'Label'

# Encode the labels
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Separating out the features and target
features = data.drop(columns=[target_column, 'ID', 'Weight']).columns  # Exclude "Weight" from features
X = data.loc[:, features].values
y = data.loc[:, target_column].values
weights = data["Weight"].values  # Get the weights

imputer = SimpleImputer(missing_values=np.nan, strategy='median')
# imputer = KNNImputer()
# imputer = IterativeImputer(verbose=2)

# imputer = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0.0)
imputed_x = imputer.fit_transform(X)

# Split data into training and testing sets, including weights
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    imputed_x, y, weights, test_size=0.2, random_state=42
    # X, y, weights, test_size=0.2, random_state=42
)


################## Hyperparamater with Optuna ############################ 
# ## Define objective function
# def objective(trial):
#     # Suggest values for hyperparameters
#     n_estimators = trial.suggest_int("n_estimators", 10, 200, log=True)
#     max_depth = trial.suggest_int("max_depth", 2, 32)
#     min_samples_split = trial.suggest_int("min_samples_split", 2, 10)
#     min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 10)

#     # Create and fit random forest model
#     model = RandomForestRegressor(
#     n_estimators=n_estimators,
#     max_depth=max_depth,
#     min_samples_split=min_samples_split,
#     min_samples_leaf=min_samples_leaf,
#     random_state=42,
#     )
#     model.fit(X_train, y_train)

#     # Make predictions and calculate RMSE
#     y_pred = model.predict(X_test)
#     rmse = np.sqrt(mean_squared_error(y_test, y_pred))
#     mae = mean_absolute_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)

#     # Return MAE
#     return mae

# # Create study object
# study = optuna.create_study(direction="minimize")

# # Run optimization process
# study.optimize(objective, n_trials=20, show_progress_bar=True)

# print("Best trial:", study.best_trial)
# print("Best hyperparameters:", study.best_params)
################## End Hyperparamater with Optuna ############################ 

##############################################

# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, f1_score
# from sklearn.preprocessing import LabelEncoder
# import numpy as np
import matplotlib.pyplot as plt


# 1. Initialize the Random Forest classifier
# Simple Impute Best hyperparameters: {'n_estimators': 122, 'max_depth': 26, 'min_samples_split': 3, 'min_samples_leaf': 4}
# Simple Impute Median Best hyperparameters: {'n_estimators': 153, 'max_depth': 32, 'min_samples_split': 6, 'min_samples_leaf': 4}

rf_classifier = RandomForestClassifier(
    # n_estimators=153,
    # max_depth=32,
    # min_samples_split=6,
    # min_samples_leaf=4,
    random_state=42,
    n_jobs=-1,
    verbose=2
)

# 2. Train the model with sample weights
rf_classifier.fit(X_train, y_train, sample_weight=weights_train)

# 3. Make predictions on the test set
y_pred = rf_classifier.predict(X_test)

# 4. Calculate weighted accuracy and F1-score
weighted_accuracy = accuracy_score(y_test, y_pred, sample_weight=weights_test)
weighted_f1 = f1_score(y_test, y_pred, average="weighted", sample_weight=weights_test)

print(f"Weighted Accuracy: {weighted_accuracy:.4f}")
print(f"Weighted F1-Score: {weighted_f1:.4f}")


print(classification_report(y_test, y_pred)) 

##############################################

# --- Calculate per-class accuracy ---

cm = confusion_matrix(y_test, y_pred, sample_weight=weights_test)
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Create a DataFrame for better readability
class_accuracy_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Accuracy': class_accuracies
})

print("\nPer-Class Accuracy:")
print(class_accuracy_df)

################## Importance Features ############################
# Built-in feature importance (Gini Importance)
importances = rf_classifier.feature_importances_
feature_imp_df = pd.DataFrame({'Feature': features, 'Gini Importance': importances}).sort_values('Gini Importance', ascending=False) 
print(feature_imp_df)

# Create a bar plot for feature importance
plt.figure(figsize=(8, 4))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Gini Importance')
plt.title('Feature Importance - Gini Importance')
plt.gca().invert_yaxis()  # Invert y-axis for better visualization
plt.show()
################## End Importance Features ############################

##############################################
import pickle
import gzip

# save the iris classification model as a pickle file
model_pkl_file = "ddos-detection.pkl"  

# with open(model_pkl_file, 'wb') as file:
with gzip.open(model_pkl_file, 'wb') as file:  
    pickle.dump(rf_classifier, file)

