# MLflow Setup and Screenshots

This document will contain instructions on setting up MLflow and screenshots of the MLflow UI.

## Setup

### Local Tracking Server Setup

1.  **Navigate to your project's root directory** in the terminal.
    ```bash
    cd path/to/your/Assignmnet#04
    ```

2.  **Start the MLflow tracking server:**
    This command tells MLflow to store run data in a local SQLite database (`mlflow.db`) and artifacts in a local directory (`mlruns`).
    ```bash
    mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
    ```
    If port 5000 is busy, you can change it (e.g., `--port 5001`).

3.  **Configure your Training Environment (Notebook/Script):**
    At the beginning of your Jupyter notebook or Python training script, set the tracking URI to point to your running server:
    ```python
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000") # Or the port you chose
    ```
    Alternatively, you can set the environment variable `MLFLOW_TRACKING_URI=http://localhost:5000` before launching your Jupyter server or running your script.

### Using `mlflow ui` (Simpler local alternative)
If you prefer not to run a separate server, `mlflow ui` can be run from your project root. It will use local file storage by default (`./mlruns` and a local `mlflow.db` might be created within `mlruns` or `.mlflow_ui.db`).
```bash
cd path/to/your/Assignmnet#04
mlflow ui --port 5000
```
In this case, your notebook usually doesn't need `mlflow.set_tracking_uri()` if it's running from the same directory structure, as MLflow will default to local tracking.

## Tracked Metrics & Parameters (Examples)

The following should be logged from your training script/notebook within an `mlflow.start_run()` block:

-   **Model Performance:** Accuracy, Precision, Recall, F1-Score, ROC-AUC Score (for both train and test sets).
-   **Training Information:** Training Time.
-   **Model Parameters:** Algorithm Type, Hyperparameters (e.g., `C`, `solver` for Logistic Regression; `n_estimators`, `max_depth` for Random Forest).
-   **Tags:** Dataset version, run descriptions, feature engineering steps.
-   **Model Artifact:** The trained model file itself.
-   **Other Artifacts (Optional):** Plots like confusion matrix, feature importance.

## MLflow UI Access

If using `mlflow server` or `mlflow ui` as described above, the UI can typically be accessed at: `http://localhost:5000`

## Example Experiment Tracking Code (for your Notebook/Script)

```python
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression # Or your chosen model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score # etc.
import pandas as pd # Assuming you load data with pandas
import numpy as np # For data manipulation

# 0. Set Tracking URI (if using a separate mlflow server)
mlflow.set_tracking_uri("http://localhost:5000")

# 1. Set Experiment
experiment_name = "CreditCardFraudDetection"
mlflow.set_experiment(experiment_name)

# Dummy data for example structure - replace with your actual data loading and splitting
# X = pd.DataFrame(np.random.rand(100, 5), columns=[f'feature_{i}' for i in range(5)])
# Y = pd.Series(np.random.randint(0, 2, 100))
# X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)


# 2. Start an MLflow Run
with mlflow.start_run(run_name="LogisticRegression_Run_1"): # Optional: give your run a name
    # --- Your actual model training code would go here ---
    # Example: model = LogisticRegression(solver='liblinear', C=1.0)
    # model.fit(X_train, Y_train)
    # Y_pred_test = model.predict(X_test)
    # Y_pred_train = model.predict(X_train)
    # accuracy_test = accuracy_score(Y_test, Y_pred_test)
    # precision_test = precision_score(Y_test, Y_pred_test)
    # --- End of your training code example ---

    # Replace with your actual model and variables
    model = LogisticRegression() # Placeholder
    accuracy_test = 0.9 # Placeholder
    precision_test = 0.85 # Placeholder

    # 3. Log Parameters (adjust to your model's actual parameters)
    mlflow.log_param("model_type", model.__class__.__name__)
    if hasattr(model, 'solver'):
        mlflow.log_param("solver", model.solver)
    if hasattr(model, 'C'):
        mlflow.log_param("C", model.C)
    # Add more relevant parameters for your specific model

    # 4. Log Metrics
    mlflow.log_metric("test_accuracy", accuracy_test)
    mlflow.log_metric("test_precision", precision_test)
    # ... log other metrics: recall, f1, roc_auc for test set
    # ... consider logging train set metrics as well for overfitting detection

    # 5. Log Tags
    mlflow.set_tag("dataset_version", "v1.0_creditcard.csv")
    mlflow.set_tag("description", "Baseline logistic regression model from notebook.")
    mlflow.set_tag("developer", "YourName")

    # 6. Log Model
    # The "model" directory here will contain the model.pkl and other MLflow model info
    mlflow.sklearn.log_model(sk_model=model, artifact_path="model")

    # 7. Log Other Artifacts (Optional)
    # Example: if you generate a feature importance plot and save it as "fi.png"
    # mlflow.log_artifact("fi.png", artifact_path="plots")

    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print(f"Model logged to: {mlflow.get_artifact_uri('model')}")

    [Screenshots of MLflow UI will be added after training the model]

    mlflow.set_tag("dataset_version", "v1.0")
    mlflow.set_tag("description", "Baseline logistic regression model")

# The run ends automatically when exiting the 'with' block

```

## Best Practices

1.  Always use meaningful experiment names (e.g., "CreditCardFraud_LogisticReg", "CreditCardFraud_RandomForest").
2.  Log all relevant parameters and metrics for reproducibility and comparison.
3.  Log the trained model (`mlflow.sklearn.log_model`) so it can be versioned and retrieved.
4.  Use tags for better organization (e.g., dataset version, type of run like "baseline" or "tuning").
5.  Document preprocessing steps if they are not part of a scikit-learn pipeline that gets logged with the model.

## Deployment Integration

MLflow tracking information is used to:
1.  Select the best performing model from various runs.
2.  Track model versions. The MLflow Model Registry can be used for more advanced versioning and stage management (e.g., "Staging", "Production").
3.  Potentially load models in your backend directly from MLflow if desired (more advanced setup).

## Screenshots

[Screenshots of MLflow UI will be added after successfully running the training notebook with MLflow logging and viewing the results in the UI.]
