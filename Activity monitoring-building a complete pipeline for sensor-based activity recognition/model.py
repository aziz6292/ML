import os
import numpy as np
from scipy.stats import skew, kurtosis
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from scipy.stats import entropy
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.preprocessing import MinMaxScaler

# Set the paths to the train and test folders
train_folder = 'training'
test_folder = 'testing'


# Function to calculate first-order mean
def first_order_mean(data):
    diff_data = np.diff(data, axis=1)
    first_order_mean_value = np.mean(diff_data, axis=1)
    return first_order_mean_value

# Function to calculate second-order mean
def second_order_mean(data):
    diff_data = np.diff(data, axis=1, n=2)
    second_order_mean_value = np.mean(diff_data, axis=1)
    return second_order_mean_value

def spectral_entropy(data):
    power_spectrum = np.square(np.abs(np.fft.fft(data)))
    power_spectrum /= np.sum(power_spectrum, axis=1, keepdims=True)
    spectral_entropy = entropy(power_spectrum, axis=1)
    return spectral_entropy

def norm_first_order_mean(data):
    norm_fdm = first_order_mean(data)
    return norm_fdm

def norm_second_order_mean(data):
    norm_sdm = second_order_mean(data)
    return norm_sdm

# Function to extract features from a single data file
def extract_features(data):
    # Example: Extracting max, min, mean, and standard deviation features
    features = []
    features.append(np.max(data, axis=1))
    features.append(np.min(data, axis=1))
    features.append(np.mean(data, axis=1))
    features.append(np.std(data, axis=1))
    
    # Calculate zero crossing
    zero_crossing = np.sum(np.diff(np.sign(data), axis=1) != 0, axis=1)
    features.append(zero_crossing)
    print("Zero Crossing Feature shape: ", zero_crossing.shape)

    # Calculate spectral energy
    spectral_energy = np.sum(np.abs(np.fft.fft(data))**2, axis=1)
    features.append(spectral_energy)
    
    # Calculate percentile 20
    percentile_20 = np.percentile(data, 20, axis=1)
    features.append(percentile_20)

    # Calculate percentile 50
    percentile_50 = np.percentile(data, 50, axis=1)
    features.append(percentile_50)

    # Calculate percentile 80
    percentile_80 = np.percentile(data, 80, axis=1)
    features.append(percentile_80)

    # Calculate first-order mean
    first_order_mean_value = first_order_mean(data)
    features.append(first_order_mean_value)
    
    # Calculate second-order mean
    second_order_mean_value = second_order_mean(data)
    features.append(second_order_mean_value)

    # Calculate interquartile range
    interquartile_range = np.percentile(data, 75, axis=1) - np.percentile(data, 25, axis=1)
    features.append(interquartile_range)

    # Calculate skewness
    skewness = skew(data, axis=1)
    features.append(skewness)

    # Calculate kurtosis
    kurtosis_value = kurtosis(data, axis=1)
    features.append(kurtosis_value)

    # Calculate spectral entropy
    
    spectral_entropy_value = spectral_entropy(data)
    features.append(spectral_entropy_value)

    # Norm of first order mean
    norm_first_order_mean_value = norm_first_order_mean(data)
    features.append(norm_first_order_mean_value)

    # Norm of second order mean
    norm_second_order_mean_value = norm_second_order_mean(data)
    features.append(norm_second_order_mean_value)

    #covariance
    reshaped_data = data.reshape(data.shape[0], -1)
    autocovariance = np.zeros((data.shape[0], data.shape[2], data.shape[2]))
    for i in range(data.shape[0]):
        autocovariance[i] = np.cov(reshaped_data[i].T)
    autocovariance_flat = autocovariance[:, np.triu_indices(data.shape[2])]
    autocovariance_reshape = autocovariance_flat.reshape(data.shape[0], -1)
    autocovariance_3d = autocovariance_reshape[:, :3]
    features.append(autocovariance_3d)

    print("Features length: ", len(features))
    concatenated_fatures = np.concatenate(features, axis=1)
    print("Concatenated features shape: ", concatenated_fatures.shape)
    return(concatenated_fatures)

# Load and extract features from the train data
train_files = ['trainAccelerometer.npy', 'trainGravity.npy', 'trainGyroscope.npy',
               'trainJinsAccelerometer.npy', 'trainJinsGyroscope.npy',
               'trainLinearAcceleration.npy', 'trainMagnetometer.npy',
               'trainMSAccelerometer.npy', 'trainMSGyroscope.npy']
train_data = []
for file in train_files:
    file_path = os.path.join(train_folder, file)
    data = np.load(file_path)
    print("individual train file size: ", data.shape)
    extracted_features = extract_features(data)
    print("extracted feature shape: ", extracted_features.shape)
    train_data.append(extracted_features)
train_data = np.array(train_data)
print("X_train shape without concatenation: ", train_data.shape)
X_train = np.concatenate(train_data, axis=1)

print("List of features size: ", len(train_data))
print("final size of all x_train dataset: ", X_train.shape)
# Load labels from the train data
train_labels_file = os.path.join(train_folder, 'trainLabels.npy')
y_train = np.load(train_labels_file)

# Load and extract features from the test data
test_files = ['testAccelerometer.npy', 'testGravity.npy', 'testGyroscope.npy',
              'testJinsAccelerometer.npy', 'testJinsGyroscope.npy',
              'testLinearAcceleration.npy', 'testMagnetometer.npy',
              'testMSAccelerometer.npy', 'testMSGyroscope.npy']
test_data = []
for file in test_files:
    file_path = os.path.join(test_folder, file)
    data = np.load(file_path)
    test_data.append(extract_features(data))
X_test = np.concatenate(test_data, axis=1)

# Load labels from the test data
test_labels_file = os.path.join(test_folder, 'testLabels.npy')
y_test = np.load(test_labels_file)


# Handle missing values using imputation
imputer = SimpleImputer(strategy='mean')
X_train_imputed = imputer.fit_transform(X_train)

# Handle missing values using imputation for the test data
X_test_imputed = imputer.transform(X_test)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# Create and train the RandomForestClassifier
model = RandomForestClassifier()
model.fit(X_train_scaled, y_train)

# Predict using the trained model
y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)  

# Evaluate the model using random forest
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

print("Train Accuracy Random Forest: {:.2f}%".format(train_accuracy * 100))
print("Test Accuracy Random Forest: {:.2f}%".format(test_accuracy * 100))

# Random Forest - Confusion Matrix
rf_confusion_matrix = confusion_matrix(y_test, y_test_pred)
print("Random Forest Confusion Matrix:")
print(rf_confusion_matrix)

# Random Forest - Average F1 Score
rf_f1_score = f1_score(y_test, y_test_pred, average='weighted')
print("Random Forest Average F1 Score: {:.2f}".format(rf_f1_score))



#-----------------------------------------------------------------------
print("----------------------------------------------------------------")

from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import GridSearchCV

# Feature Scaling
#standard scaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.fit_transform(X_test_imputed)
#print(X_test_scaled)

# Feature Selection using SelectKBest
k_best = SelectKBest(k=275)  # Select top 275 features
X_train_selected = k_best.fit_transform(X_train_scaled, y_train)
X_test_selected = k_best.transform(X_test_scaled)

# Hyperparameter Tuning for SVM
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
    'gamma': ['scale', 'auto']
}
svm_model = SVC()
grid_search_svm = GridSearchCV(svm_model, param_grid=param_grid_svm, cv=5)
grid_search_svm.fit(X_train_selected, y_train)

# Train the SVM model with best parameters
best_svm_model = grid_search_svm.best_estimator_
best_svm_model.fit(X_train_selected, y_train)

# Predict labels for the test data
y_train_pred_svm = best_svm_model.predict(X_train_selected)
y_test_pred_svm = best_svm_model.predict(X_test_selected)

# Evaluate the model using SVM
train_accuracy_svm = accuracy_score(y_train, y_train_pred_svm)
test_accuracy_svm = accuracy_score(y_test, y_test_pred_svm)

print("Train Accuracy SVM: {:.2f}%".format(train_accuracy_svm * 100))
print("Test Accuracy SVM: {:.2f}%".format(test_accuracy_svm * 100))

# SVM - Confusion Matrix
svm_confusion_matrix = confusion_matrix(y_test, y_test_pred_svm)
print("SVM Confusion Matrix:")
print(svm_confusion_matrix)

# SVM - Average F1 Score
svm_f1_score = f1_score(y_test, y_test_pred_svm, average='weighted')
print("SVM Average F1 Score: {:.2f}".format(svm_f1_score))