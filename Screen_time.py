import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit 
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression

screen = pd.read_csv("Indian_Kids_Screen_Time.csv")

# split data using Age columun
screen["Age_cat"]= pd.cut(
    screen["Age"],
    bins=[0,8,10,12,14,16,18],
    labels=[1,2,3,4,5,6]
)

# Now split data set
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_data, test_data in split.split(screen, screen["Age_cat"]):
    train_set = screen.loc[train_data].drop("Age_cat",axis=1)
    test_set = screen.loc[test_data].drop("Age_cat", axis=1)

screen = train_set.copy()

# Remove missing target rows
screen = screen.dropna(subset=["Health_Impacts"])

# seperate lable and features
screen_label = screen["Health_Impacts"].copy()
screen = screen.drop("Health_Impacts", axis=1)

# seperate numerical and categorical column
num_attributes = screen.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_attributes = screen.select_dtypes(include=["object","bool"]).columns.tolist()

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Full pipeline
full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attributes),
    ("cat", cat_pipeline, cat_attributes)
])

screen_prepared = full_pipeline.fit_transform(screen)

# Train random forest classifier
random_clf = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
random_clf.fit(screen_prepared, screen_label)

# Training predictions
preds = random_clf.predict(screen_prepared)

# Accuracy
train_acc = accuracy_score(screen_label, preds)
print(f"Training Accuracy: {train_acc:.4f}")


# --------------------------------------------------


# Prepare test data
screen_test = test_set.copy()

# Remove missing target rows
screen_test = screen_test.dropna(subset=["Health_Impacts"])

# Separate features and label
y_test = screen_test["Health_Impacts"].copy()
screen_test = screen_test.drop("Health_Impacts", axis=1)

# Transform test data 
screen_test_prepared = full_pipeline.transform(screen_test)

test_preds = random_clf.predict(screen_test_prepared)

# Test Accuracy
test_acc = accuracy_score(y_test, test_preds)
print(f"Test Accuracy: {test_acc:.4f}")