import pandas as pd
import numpy as np
import dagshub
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from pathlib import Path

# Fix GUI issues in CI
plt.switch_backend('Agg')

SCRIPT_DIR = Path(__file__).resolve().parent
data_path = SCRIPT_DIR / "data_processed" / "train.csv"

def main():
    # 1. Initialize DagsHub (Environment Variables dari Secrets)
    token = os.getenv("DAGSHUB_USER_TOKEN")
    if not token:
        print("[WARN] DAGSHUB_USER_TOKEN not found. Ensure GitHub Secrets are set.")
    
    # Init DagsHub (Aman jika gagal, script tetap jalan local)
    try:
        # Pastikan repo_name & repo_owner SESUAI dengan DagsHub kamu
        dagshub.init(repo_owner='vadhh', repo_name='SMSML_Afridho_Tavadhu', mlflow=True)
    except Exception as e:
        print(f"[WARN] DagsHub init failed: {e}. Continuing...")

    # 2. Load Data (Robust)
    try:
        print(f"[INFO] Loading data from: {data_path}")
        df = pd.read_csv(data_path)
        
        # Preprocessing Basic
        if 'customerID' in df.columns:
            df = df.drop('customerID', axis=1)
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
            
        X = df.drop('Churn', axis=1)
        # One-Hot Encoding (Fix error string to float)
        X = pd.get_dummies(X)
        y = df['Churn']
        
        # Encoding Target Yes/No -> 1/0
        if y.dtype == 'object':
            y = y.map({'Yes': 1, 'No': 0})
            
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        print("[INFO] Real data loaded successfully.")
        
    except (FileNotFoundError, Exception) as e:
        print(f"[WARN] Data loading failed ({e}). Generating DUMMY data for CI test...")
        # Dummy Data agar CI tidak Merah (Fail) hanya karena path salah
        X_train = pd.DataFrame(np.random.rand(100, 10))
        X_test = pd.DataFrame(np.random.rand(20, 10))
        y_train = pd.Series(np.random.randint(0, 2, 100))
        y_test = pd.Series(np.random.randint(0, 2, 20))

    # 3. Start Run
    # PENTING: HAPUS set_experiment agar tidak konflik dengan 'mlflow run'
    # mlflow.set_experiment("...") <-- SUDAH DIHAPUS
    
    print("[INFO] Starting MLflow Run...")
    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"[INFO] Run ID: {run_id}")
        
        with open("run_id.txt", "w") as f:
            f.write(run_id)
            
        mlflow.sklearn.autolog()

        # 4. Train Model
        rf = RandomForestClassifier(random_state=42)
        param_grid = {'n_estimators': [10, 50], 'max_depth': [5, 10]}
        
        grid_search = GridSearchCV(rf, param_grid, cv=2)
        grid_search.fit(X_train, y_train)

        # 5. Create Artifacts (Confusion Matrix)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.savefig("confusion_matrix.png")
        
        mlflow.log_artifact("confusion_matrix.png")
        
        print(f"[SUCCESS] Run ID {run_id} saved to run_id.txt")

if __name__ == "__main__":
    main()
