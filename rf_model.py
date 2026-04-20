import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from scipy import stats

def main():
    # Load the 1250 dataset which contains risky to safe dose level data
    df = pd.read_csv('smart_vet_dose_1250.csv')
    
    X = df.drop("Outcome", axis=1)
    y = df["Outcome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # Train Random Forest Classifier
    rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, rf_pred)
    precision = precision_score(y_test, rf_pred)
    recall = recall_score(y_test, rf_pred)
    f1 = f1_score(y_test, rf_pred)

    print("--- Random Forest Model Evaluation ---")
    print(f"Accuracy:  {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall:    {recall:.2f}")
    print(f"F1 Score:  {f1:.2f}")

    # Calculate recommended safe dosage level
    safe_data = df[df['Outcome'] == 1]
    safe_doses = safe_data['Dose_mg_per_kg']
    
    mean_dose = safe_doses.mean()
    std_dose = safe_doses.std()
    n = len(safe_doses)
    
    # 95% Confidence Interval for the mean
    z = 1.96
    margin_of_error = z * (std_dose / np.sqrt(n))
    
    lower_bound = mean_dose - margin_of_error
    upper_bound = mean_dose + margin_of_error
    
    print("\n--- Pharmacokinetic Validation ---")
    print(f"Recommended safe dosage interval: {lower_bound:.2f} to {upper_bound:.2f} mg/kg")
    
if __name__ == "__main__":
    main()
