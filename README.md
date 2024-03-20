# Toy Machine ML Report

## Overview
The machine learning problem we are addressing is a type of Supervised Learning called Classification. This approach uses labeled data to train algorithms, enabling them to predict outcomes or categorize data. In our case, the target label (y) consists of either 1's or 0's. Our dataset is relatively small, allowing us to use batch learning, which trains the model on all the data at once. In total, the dataset contains 6 columns and 11,000 rows.

## Getting the Data
The first step I took was to load the data from the CSV file into a dataframe. Then, I split the dataset into two parts: the training set and the testing set, using scikit-learn's `train_test_split` function. The first 5 columns (features) were assigned to X, and the last column (target label) was assigned to y. The testing set was created by selecting 1000 random rows (samples) from the dataset, with a seed set to 42.

```python
# Load in the data from the CSV file and sets a name for each column
def load_toy_machine_data():
   url = "dataSet.csv"
   return pd.read_csv(url, names=["c1" , "c2", "c3", "c4", "c5", "c6"])
toy_machine_data = load_toy_machine_data()

X = toy_machine_data.loc[:,"c1":"c5"] # Feature columns
y = toy_machine_data["c6"] # Target label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42)
```

## Preparing the Data
Preparing the data involved cleaning it. The first step was to get a feel for the data using the `.head()` command to examine the first 5 rows of the dataset, revealing that all feature column values ranged from 0.0 to 1.0, indicating no need for feature scaling. Additionally, no columns contained strings, eliminating the need for encoding. Running the `.info()` command showed that each column's datatype and there were no missing values in the dataset. The `.describe()` command provided various statistics, from the mean to min/max values. Plotting the data with histograms did not reveal any insights due to unnamed feature columns, these steps showed that the data was already clean and ready for use.

```python
print(toy_machine_data.head())
print(toy_machine_data.info())
print(toy_machine_data.describe())
toy_machine_data.hist(bins=50, figsize=(12, 8))
plt.show()
```

## Select and Train a Model
I tested four different models using Scikit-learn: Random Forest Classifier, Support Vector Machine, Logistic Regression, and Decision Trees. Due to Logistic Regression's lower base accuracy compared to the other models, I scraped it. After evaluating the remaining models, the Random Forest and Support Vector Machine showed the highest accuracy. The accuracy was calculated using the cross-validation score.

## Grid Search and Cross-Validation
Each model was trained using 5-fold cross-validation, meaning the data was divided into five parts, with four being used for training and one for testing. To fine-tune the models, I used Scikit-Learn's GridSearchCV, which automates the search for the best combination of hyperparameters for each model. A dictionary of hyperparameters relevant to each specific model was created, with values adjusted to achieve the best accuracy for the dataset. Essentially, the process involves creating a model and then fine-tuning its hyperparameters with GridSearchCV before training it on the dataset again.

```python
# Random Forest Parameters
params_rf = {
   'n_estimators': [800, 900, 1000], # Number of trees in the forest
   'max_depth': [45], # Max depth of a tree
   'min_samples_split': [2],
   'min_samples_leaf': [1]
}

# Support Vector Machine Parameters
params_svm = {
   'C': [1100, 1150, 1200] # Regularization parameter
}

# Decision Tree Parameters
params_dt = {
   'max_depth': [11,12,13],
   'min_samples_split': [2,3],
   'min_samples_leaf': [2,3]
}

def model_training(model_name, model, params):
   grid_search = GridSearchCV(model, params, cv=5, scoring="accuracy")
   grid_search.fit(X_train, y_train)

   print(f"Best parameters for {model_name}:", grid_search.best_params_)
   print(f"Best cross-validation score for {model_name}:",grid_search.best_score_)

   return grid_search.best_estimator_

random_forest = RandomForestClassifier()
tuned_rf = model_training("Random Forest", random_forest, params_rf)

support_vector_machine = SVC()
tuned_svm= model_training("Support Vector Machine", support_vector_machine, params_svm)

decision_tree = DecisionTreeClassifier()
tuned_dt = model_training("Decision Tree", decision_tree, params_dt)
```

## Evaluate The Model on The Test Set
After fine-tuning the hyperparameters, the Support Vector Machine was the most accurate for the test set, getting the highest cross-validation scores, peaking at 91%. In contrast, the Random Forest and Decision Tree models peaked at 87.3% and 79.5%, respectively. The next step was to evaluate the Support Vector Machine model, with its tuned hyperparameters, on the test set. The model's predictions for the test set's feature columns were then compared to the actual target label values.

```python
y_pred = tuned_svm.predict(X_test)
```

## Results and Insights
To evaluate the model's performance, I used Scikit-learn's metrics to calculate accuracy, precision, recall, F1 Score, and to generate the confusion matrix. The model's accuracy is at 91%, meaning that it makes correct predictions 91% of the time. The model's precision is 91%, which means it’s pretty accurate when it says something is positive. And with a recall of 91%, it’s good at spotting most of the things that are actually positive. The F1-score, at 91%, shows that the model is balanced with regard to precision and recall.

**Metrics Results**
- Accuracy: 0.91
- Precision: 0.91
- Recall: 0.91
- F1-score: 0.91

```python
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=tuned_svm.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".3g")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
```

The confusion matrix showed the model correctly predicted '0' 422 times and '1' 488 times. However, it incorrectly predicted '1' when it was actually '0' 42 times, and '0' when it was actually '1' 48 times.

<img width="630" alt="Screenshot 2024-03-20 at 2 46 24 PM" src="https://github.com/jawadrada/Toy-Machine-ML/assets/103535961/8571ddbf-8112-4b17-889e-3046515b6545">
