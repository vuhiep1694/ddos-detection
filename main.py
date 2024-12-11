import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import pickle
import gzip
import os
import gdown


################## INPUT HERE ########################
# Load the dataset
# Assuming the dataset is a CSV file
data = pd.read_csv('/Users/lap16109/Downloads/ddos/new-test.csv')

##############################################
### Don't change anything from here
data.replace([np.inf, -np.inf], np.nan, inplace=True)

# Encode the labels
target_column = 'Label'
label_encoder = LabelEncoder()
data[target_column] = label_encoder.fit_transform(data[target_column])

# Separating out the features and target
features = data.drop(columns=[target_column, 'ID', 'Weight']).columns  # Exclude "Weight" from features
X = data.loc[:, features].values
y = data.loc[:, target_column].values
weights = data["Weight"].values  # Get the weights

#download model from gg drive : https://drive.google.com/file/d/1NgHVY-o7s4Ru_lDYpYELATsVF1AxtEMe/view?usp=sharing
model_path=os.getcwd()+'/model'
if not os.path.exists(model_path):
	os.makedirs(model_path)

model_pkl_file = model_path+'/ddos-detection.pkl'
if not os.path.isfile(model_pkl_file):
    print('Downloading model from google drive, please wait....')
    url = 'https://drive.google.com/uc?id=1NgHVY-o7s4Ru_lDYpYELATsVF1AxtEMe'
    gdown.download(url, model_pkl_file, quiet=False)

# load model from pickle file
with gzip.open(model_pkl_file, 'rb') as file:  
    model = pickle.load(file)

# evaluate model 
y_predict = model.predict(X)

# check results
print('===================== classification_report ======================')
print(classification_report(y, y_predict, sample_weight=weights))

# classify
print('===================== Per-Class Accuracy ==========================')
cm = confusion_matrix(y, y_predict, sample_weight=weights)
class_accuracies = cm.diagonal() / cm.sum(axis=1)

# Create a DataFrame for better readability
class_accuracy_df = pd.DataFrame({
    'Class': label_encoder.classes_,
    'Accuracy': class_accuracies
})

print("\nPer-Class Accuracy:")
print(class_accuracy_df)