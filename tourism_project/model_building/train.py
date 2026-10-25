import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score

def main():
    # Load datasets
    train_path = 'tourism_project/model_building/train.csv'
    test_path = 'tourism_project/model_building/test.csv'

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Error: Splits not found.")
        return

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    # Handle Missing Values (simple imputation)
    for df in [train_df, test_df]:
        # Impute numeric columns with median
        num_cols = df.select_dtypes(include=['int64', 'float64']).columns
        for col in num_cols:
            df[col] = df[col].fillna(df[col].median())
        # Impute categorical columns with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            df[col] = df[col].fillna(df[col].mode()[0])

    target = 'ProdTaken'
    X_train = train_df.drop(columns=[target])
    y_train = train_df[target]
    X_test = test_df.drop(columns=[target])
    y_test = test_df[target]

    # Detect column types
    num_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_features = X_train.select_dtypes(include=['object']).columns.tolist()

    # Column Transformer
    preprocessor = make_column_transformer(
        (StandardScaler(), num_features),
        (OneHotEncoder(handle_unknown='ignore'), cat_features)
    )

    # Pipeline with XGBoost
    pipeline = make_pipeline(
        preprocessor,
        xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    )

    # Hyperparameter tuning
    param_grid = {
        'xgbclassifier__max_depth': [3, 5],
        'xgbclassifier__n_estimators': [50, 100]
    }

    # Explicitly point to the local MLflow server running in GHA background
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    mlflow.set_experiment("Tourism_Prediction")
    
    with mlflow.start_run():
        grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
        grid.fit(X_train, y_train)

        best_model = grid.best_estimator_
        y_pred = best_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Log params and metrics
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.sklearn.log_model(best_model, "model")

        print(f"Best Hyperparameters: {grid.best_params_}")
        print(f"Accuracy on Test Set: {accuracy:.4f}")
        print(classification_report(y_test, y_pred))

        # Save model locally
        os.makedirs("tourism_project/deployment", exist_ok=True)
        joblib.dump(best_model, 'tourism_project/deployment/model.pkl')
        print("Model saved to tourism_project/deployment/model.pkl")

if __name__ == '__main__':
    main()
