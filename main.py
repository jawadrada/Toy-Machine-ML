import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# Load in the data from the csv file and sets a name for each column
def load_toy_machine_data():
    url = "dataSet.csv"
    return pd.read_csv(url, names=["c1" , "c2", "c3", "c4", "c5", "c6"])
toy_machine_data = load_toy_machine_data()

print(toy_machine_data.head())
print(toy_machine_data.info())
print(toy_machine_data.describe())
toy_machine_data.hist(bins=50, figsize=(12, 8))
plt.show()

X = toy_machine_data.loc[:,"c1":"c5"] # Feature columns
y = toy_machine_data["c6"] # target label
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1000, random_state=42)

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
    print(f"Best cross-validation score for {model_name}:", grid_search.best_score_)

    return grid_search.best_estimator_ 

random_forest = RandomForestClassifier()
tuned_rf = model_training("Random Forest", random_forest, params_rf)

support_vector_machine = SVC()
tuned_svm= model_training("Support Vector Machine", support_vector_machine, params_svm)

decision_tree = DecisionTreeClassifier()
tuned_dt = model_training("Decision Tree", decision_tree, params_dt)

y_pred = tuned_rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy}")
print(classification_report(y_test, y_pred))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_support_vector_machine.classes_)
disp.plot(cmap=plt.cm.Blues, values_format=".3g")
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Hidden Test Set
url1 = "hiddenTestSet.csv"
hidden_data = pd.read_csv(url1, names=["c1" , "c2", "c3", "c4", "c5"])
X_hidden = hidden_data.loc[:,"c1":"c5"]
y_pred_hidden = tuned_svm.predict(X_hidden)
print(y_pred_hidden)
