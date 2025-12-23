import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
import dagshub
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

# Ensure no GUI issues
plt.switch_backend('Agg')

SCRIPT_DIR = Path(__file__).resolve().parent

# Go up two levels: MLProject -> Workflow-CI -> Root
PROJECT_ROOT = SCRIPT_DIR.parent.parent 

data_path = PROJECT_ROOT / "data_processed" / "train.csv"

def main():
    # 1. Initialize DagsHub/MLflow
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if not token:
        raise ValueError("CRITICAL: DAGSHUB_USER_TOKEN environment variable is missing or empty. Check GitHub Secrets.")
    
    dagshub.init(repo_owner='vadhh', repo_name='SMSML_Afridho_Tavadhu', mlflow=True)
    
    # 2. Load Data (Assumes data is present or mounted)
    try:
        df = pd.read_csv(data_path) # Adjust path if needed or use relative
        X = df.drop('Churn', axis=1)
        y = df['Churn']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    except FileNotFoundError:
        print("[WARN] Data not found, generating dummy data for CI test...")
        X_train, X_test = pd.DataFrame(np.random.rand(100, 10)), pd.DataFrame(np.random.rand(20, 10))
        y_train, y_test = pd.Series(np.random.randint(0, 2, 100)), pd.Series(np.random.randint(0, 2, 20))

    # 3. Start Run
    mlflow.set_experiment("CI_CD_Pipeline_Experiment")
    
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[INFO] Run ID: {run_id}")
        
        # Write Run ID to file for GitHub Actions to read
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        mlflow.sklearn.autolog()

        # 4. Train
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 50], 'max_depth': [5, 10]}
        grid_search = GridSearchCV(rf, param_grid, cv=2)
        grid_search.fit(X_train, y_train)

        # 5. Create Artifacts (Confusion Matrix)
        y_pred = grid_search.best_estimator_.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True)
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
        print(f"[SUCCESS] Model trained and logged. Run ID saved to run_id.txt")

if __name__ == "__main__":
    main()


    rf.fit(X_train, y_train)
    
    run_id = run.info.run_id
    
    with open("run_id.txt", "w") as f:
        f.write(run_id)

    print(f"Training Selesai. Run ID {run_id} disimpan ke run_id.txt")
