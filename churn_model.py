import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

matplotlib.use('Agg')

def churn_model():

    # Load the dataset
    df = pd.read_csv('WA_Fn-UseC_-Telco-Customer-Churn.csv')

    # Convert TotalCharges to numeric
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    print("-------------------------------------------------------------------")
    # Check if any nulls appeared after conversion
    print(df['TotalCharges'].isnull().sum())
    print("-------------------------------------------------------------------")

    # Drop the 11 rows with null TotalCharges
    df = df.dropna(subset=['TotalCharges'])

    # Convert Churn to 0 and 1
    df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

    # Drop customerID - it's just an identifier, not useful for prediction
    df = df.drop(columns=['customerID'])

    print("-------------------------------------------------------------------")
    # Verify
    print(df.shape)
    print(df['Churn'].value_counts())
    print("-------------------------------------------------------------------")

    # Check all categorical columns
    print("-------------------------------------------------------------------")
    print(df.select_dtypes(include='string').columns.tolist())
    print("-------------------------------------------------------------------")

    # Encode categorical variables using Label Encoding
    le = LabelEncoder()

    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 
                        'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                        'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                        'PaperlessBilling', 'PaymentMethod']

    for col in categorical_cols:
        df[col] = le.fit_transform(df[col])

    X = df.drop(columns=['Churn']).copy()
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("-------------------------------------------------------------------")
    print("Training set size:", X_train.shape)
    print("Testing set size:", X_test.shape)
    print("-------------------------------------------------------------------")
    models = {
        'Logistic Regression': LogisticRegression(max_iter=2000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBClassifier(n_estimators=100, random_state=42)
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{name} Accuracy: {accuracy * 100:.2f}%")
        print(f"{name} Classification Report:\n{classification_report(y_test, y_pred)}")
        print(f"{name} Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
        print("-------------------------------------------------------------------")

    param_grid = {
    'n_estimators': [80, 90, 100],
    'max_depth': [5, 6, 7],
    'min_samples_split': [2, 3, 4]
    }

    grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    print("-------------------------------------------------------------------")

    print(f"Best Parameters: {grid_search.best_params_}")
    print(f"Best Accuracy: {grid_search.best_score_*100:.2f}%")
    print("-------------------------------------------------------------------")
    best_model = grid_search.best_estimator_
    accuracy = best_model.score(X_test, y_test)
    print(f"Final Tuned Model Accuracy: {accuracy * 100:.2f}%")
    print("-------------------------------------------------------------------")

    dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt_model.fit(X_train, y_train)

    plt.figure(figsize=(20,8))
    plot_tree(dt_model, feature_names=X.columns, class_names=['No Churn', 'Churn'], filled=True)
    plt.title('Decision Tree Visualization')
    plt.savefig('decision_tree.png', bbox_inches='tight')
    plt.close()


    feature_importance = pd.Series(best_model.feature_importances_, index=X.columns)
    feature_importance.sort_values(ascending=False).plot(kind='bar')
    plt.title('Feature Importance')
    plt.ylabel('Importance Score')
    plt.savefig('feature_importance.png', bbox_inches='tight')
    plt.close()

if __name__ == "__main__":
    churn_model()