import pandas as pd
import numpy as np
from collections import Counter
from ucimlrepo import fetch_ucirepo

# Set random seed for reproducibility
np.random.seed(53)
used_datasets = {
    "iris": 53,
    "heart_disease": 45,
    "wine_quality": 186,
    "breast_cancer": 17,
    "adult": 2,
    "bank_marketing": 222,
    "student_performance": 320,
    "wine": 109,
    "air_quality": 360,
    "mushroom": 73
}
iris = fetch_ucirepo(id=53)
# print(iris)
# data (as pandas dataframes)
X = iris.data.features
y = iris.data.targets

# Concatenate features and targets for shuffling and splitting

data = pd.concat([X, y], axis=1)
dataset = data.to_numpy()
# print(dataset)

print("Classes in the dataset:", np.unique(y))

# Shuffle the dataset
np.random.shuffle(dataset)

# Split into training and testing sets (80% train, 20% test)
train_split = 0.8
train_size = int(train_split * len(dataset))
train_data = dataset[:train_size]
test_data = dataset[train_size:]

print(f"Total dataset size: {len(dataset)}")
print(f"Training data size: {len(train_data)}")
print(f"Test data size: {len(test_data)}")

class NaiveBayes:
    def fit(self, X, y):
        self.classes = np.unique(y)
        self.parameters = {}
        for i, cls in enumerate(self.classes):
            # Filter X and y for the current class
            X_cls = X[y == cls]
            # print(X_cls)
            # Calculate mean and variance for each feature for the current class
            # Adding a small epsilon to variance to prevent division by zero
            self.parameters[cls] = {
                'mean': X_cls.mean(axis=0),
                'variance': X_cls.var(axis=0) + 1e-6
            }
        
        # Calculate prior probabilities for each class
        self.class_priors = {cls: np.sum(y == cls) / len(y) for cls in self.classes}

    def _pdf(self, X_row, mean, variance):
        # Gaussian Probability Density Function
        exponent = -((X_row - mean)**2) / (2 * variance)
        return (1 / np.sqrt(2 * np.pi * variance)) * np.exp(exponent)

    def predict(self, X):
        predictions = [self._predict_single(x) for x in X]
        return np.array(predictions)

    def _predict_single(self, x):
        posteriors = []
        for cls in self.classes:
            prior = np.log(self.class_priors[cls])
            
            # Ensure x and parameters are float for calculation
            likelihood = np.sum(np.log(self._pdf(x.astype(float), self.parameters[cls]['mean'], self.parameters[cls]['variance'])))
            
            posterior = prior + likelihood
            posteriors.append(posterior)
        
        return self.classes[np.argmax(posteriors)]

# Separate features and labels from the full dataset
X_train = train_data[:, :-1].astype(float) 
# print(X_train)
y_train = train_data[:, -1]
X_test = test_data[:, :-1].astype(float)   
y_test = test_data[:, -1]

# Instantiate and train the model
classifier = NaiveBayes()
classifier.fit(X_train, y_train)

# Make predictions on the test set
y_pred = classifier.predict(X_test)


def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

def error_rate(y_true, y_pred):
    return 1 - accuracy(y_true, y_pred)

def precision_recall_fscore(y_true, y_pred, average='weighted', beta=1.0):
    classes = np.unique(y_true)
    precision_scores = []
    recall_scores = []
    fscore_scores = []
    
    for cls in classes:
        true_positives = np.sum((y_pred == cls) & (y_true == cls))
        false_positives = np.sum((y_pred == cls) & (y_true != cls))
        false_negatives = np.sum((y_pred != cls) & (y_true == cls))
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        fscore = ((1 + beta**2) * precision * recall) / ((beta**2 * precision) + recall) if (precision + recall) > 0 else 0
        
        precision_scores.append(precision)
        recall_scores.append(recall)
        fscore_scores.append(fscore)
        
    if average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in classes]
        total_samples = len(y_true)
        
        avg_precision = np.sum(np.array(precision_scores) * np.array(weights)) / total_samples
        avg_recall = np.sum(np.array(recall_scores) * np.array(weights)) / total_samples
        avg_fscore = np.sum(np.array(fscore_scores) * np.array(weights)) / total_samples
        
        return avg_precision, avg_recall, avg_fscore
    else:
        return np.mean(precision_scores), np.mean(recall_scores), np.mean(fscore_scores)

def specificity_score(y_true, y_pred, average='weighted'):
    classes = np.unique(y_true)
    specificity_scores = []

    for cls in classes:
        true_negatives = np.sum((y_pred != cls) & (y_true != cls))
        false_positives = np.sum((y_pred == cls) & (y_true != cls))
        
        specificity = true_negatives / (true_negatives + false_positives) if (true_negatives + false_positives) > 0 else 0
        specificity_scores.append(specificity)
        
    if average == 'weighted':
        weights = [np.sum(y_true == cls) for cls in classes]
        total_samples = len(y_true)
        return np.sum(np.array(specificity_scores) * np.array(weights)) / total_samples
    else:
        return np.mean(specificity_scores)

acc = accuracy(y_test, y_pred)
err_rate = error_rate(y_test, y_pred)
precision, recall, f1_score = precision_recall_fscore(y_test, y_pred, beta=1.0)
f2_score = precision_recall_fscore(y_test, y_pred, beta=2.0)[2]
f0_5_score = precision_recall_fscore(y_test, y_pred, beta=0.5)[2]
specificity = specificity_score(y_test, y_pred)

print(f"Accuracy of the multi-class decision tree: {acc:.2f}")
print(f"Error Rate: {err_rate:.2f}")
print(f"Precision (weighted): {precision:.2f}")
print(f"Recall (weighted): {recall:.2f}")
print(f"F1-Score (weighted): {f1_score:.2f}")
print(f"F2-Score (weighted, beta=2.0): {f2_score:.2f}")
print(f"F0.5-Score (weighted, beta=0.5): {f0_5_score:.2f}")
print(f"Specificity (weighted): {specificity:.2f}")

print("\n--- All Test Predictions ---")
for i in range(len(X_test)):
    sample_x = X_test[i]
    sample_y_true = y_test[i]
    sample_y_pred = y_pred[i]
    #if (sample_y_pred!=sample_y_true):
    print(f"Sample {i+1}: Predicted='{sample_y_pred}', Actual='{sample_y_true}'")
        
with open("naive_bayes_results.txt", "w") as f:
    # Write performance metrics
    f.write(f"Accuracy of the Naive Bayes classifier: {acc:.2f}\n")
    f.write(f"Error Rate: {err_rate:.2f}\n")
    f.write(f"Precision (weighted): {precision:.2f}\n")
    f.write(f"Recall (weighted): {recall:.2f}\n")
    f.write(f"F1-Score (weighted): {f1_score:.2f}\n")
    f.write(f"F2-Score (weighted, beta=2.0): {f2_score:.2f}\n")
    f.write(f"F0.5-Score (weighted, beta=0.5): {f0_5_score:.2f}\n")
    f.write(f"Specificity (weighted): {specificity:.2f}\n")
    
    f.write("\n--- All Test Predictions ---\n")
    for i in range(len(X_test)):
        sample_y_true = y_test[i]
        sample_y_pred = y_pred[i]
        if sample_y_pred != sample_y_true:
            f.write(f"Sample {i+1}: Predicted='{sample_y_pred}', Actual='{sample_y_true}'\n")

