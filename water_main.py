import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from imblearn.over_sampling import SMOTE
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

data = pd.read_csv(r"C:\Users\milan\OneDrive\Desktop\Water_Potability_Prediction\water_potability.csv")

#calculating mean values for each field for both categories
#for sulfate
sulfate_0 = data[data['Potability']==0]['Sulfate'].mean(skipna=True)
sulfate_1 = data[data['Potability']==1]['Sulfate'].mean(skipna=True)

#for ph
ph_0 = data[data['Potability']==0]['ph'].mean(skipna=True)
ph_1 = data[data['Potability']==1]['ph'].mean(skipna=True)

#for Trihalomethanes
Trihalo_0 = data[data['Potability']==0]['Trihalomethanes'].mean(skipna=True)
Trihalo_1 = data[data['Potability']==1]['Trihalomethanes'].mean(skipna=True)

#sulfate mean imputation
data.loc[(data['Potability'] == 0) & (data['Sulfate'].isna()), 'Sulfate'] = sulfate_0
data.loc[(data['Potability'] == 1) & (data['Sulfate'].isna()), 'Sulfate'] = sulfate_1

#ph mean imputation
data.loc[(data['Potability'] == 0) & (data['ph'].isna()), 'ph'] = ph_0
data.loc[(data['Potability'] == 1) & (data['ph'].isna()), 'ph'] = ph_1

#Trihalomethanes mean imputation
data.loc[(data['Potability'] == 0) & (data['Trihalomethanes'].isna()), 'Trihalomethanes'] = Trihalo_0
data.loc[(data['Potability'] == 1) & (data['Trihalomethanes'].isna()), 'Trihalomethanes'] = Trihalo_1

x = data.drop('Potability', axis=1)
y = data['Potability']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=24)

# Fit GMM on the minority class (Potability == 1) to generate synthetic data
gmm = GaussianMixture(n_components=2, random_state=42)

# Train the GMM on the minority class data
X_minority = x_train[y_train == 1]
gmm.fit(X_minority)

# Generate synthetic samples from GMM
synthetic_data, _ = gmm.sample(n_samples=100)  # Specify the number of synthetic samples

# Create a DataFrame for the synthetic samples
synthetic_df = pd.DataFrame(synthetic_data, columns=x.columns)

# Assign the target label (Potability == 1) to the synthetic data
synthetic_labels = np.ones(synthetic_df.shape[0])

# Combine original training data with synthetic data
X_train_Smote = np.vstack([x_train, synthetic_df])
y_train_Smote = np.hstack([y_train, synthetic_labels])

# Define the base RF model
base_RF = RandomForestClassifier(random_state=0)

# Define the RF hyperparameter 
param_grid_RF = {
    'n_estimators': [10, 30, 50],  # Number of trees in the forest.
    'criterion': ['gini', 'entropy'],  # Function to measure split quality.
    'max_depth': [2, 3, 4],  # Maximum depth of each tree.
    'min_samples_split': [2, 3, 4, 5],  # Minimum samples required to split a node.
    'min_samples_leaf': [1, 2, 3],  # Minimum samples required to be a leaf node.
    'bootstrap': [True, False]  # Whether to use bootstrap samples.
}

gridSearch_RF = GridSearchCV(estimator=base_RF, param_grid=param_grid_RF, cv=5, scoring='accuracy', n_jobs=-1)
gridSearch_RF.fit(X_train_Smote, y_train_Smote)

best_params_RF = gridSearch_RF.best_params_

best_RF = RandomForestClassifier(random_state=42, **best_params_RF)
best_RF.fit(X_train_Smote, y_train_Smote)

# Save the trained model to a file
dump(best_RF, "water_randomforest.joblib")