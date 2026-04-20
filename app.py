import os
from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)

# Train model at startup
df = pd.read_csv('smart_vet_dose_1250.csv')
X = df.drop("Outcome", axis=1)
y = df["Outcome"]
rf = RandomForestClassifier(n_estimators=150, max_depth=6, random_state=42)
rf.fit(X, y)

safe_data = df[df['Outcome'] == 1]
safe_doses = safe_data['Dose_mg_per_kg']
mean_dose = safe_doses.mean()
std_dose = safe_doses.std()
margin_of_error = 1.96 * (std_dose / np.sqrt(len(safe_doses)))
safe_lower = mean_dose - margin_of_error
safe_upper = mean_dose + margin_of_error

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        weight = float(data.get('weight', 20))
        age = float(data.get('age', 5))
        rbc = float(data.get('rbc', 6.5))
        hb = float(data.get('hb', 14))
        creatinine = float(data.get('creatinine', 1))
        glucose = float(data.get('glucose', 100))
        
        # Test doses from 0.1 to 3.0 to find the curve
        doses = np.arange(0.1, 3.1, 0.05)
        test_df = pd.DataFrame({
            "Weight": [weight]*len(doses),
            "Age": [age]*len(doses),
            "RBC": [rbc]*len(doses),
            "HB": [hb]*len(doses),
            "Creatinine": [creatinine]*len(doses),
            "Glucose": [glucose]*len(doses),
            "Dose_mg_per_kg": doses
        })
        
        probs = rf.predict_proba(test_df)[:, 1] # Probability of being Safe (1)
        
        max_prob_idx = np.argmax(probs)
        optimal_dose = doses[max_prob_idx]
        max_prob = probs[max_prob_idx]
        
        safe_indices = np.where(probs >= 0.5)[0]
        if len(safe_indices) > 0:
            patient_safe_lower = doses[safe_indices[0]]
            patient_safe_upper = doses[safe_indices[-1]]
            status = "Safe Range Verified"
        else:
            patient_safe_lower = safe_lower
            patient_safe_upper = safe_upper
            status = "High Anesthetic Risk"

        return jsonify({
            "optimal_dose": round(optimal_dose, 2),
            "safe_lower": round(patient_safe_lower, 2),
            "safe_upper": round(patient_safe_upper, 2),
            "max_prob": round(float(max_prob) * 100, 1),
            "status": status,
            "population_lower": round(safe_lower, 2),
            "population_upper": round(safe_upper, 2),
            "chart_labels": [round(d, 2) for d in doses],
            "chart_data": [round(p * 100, 2) for p in probs]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, port=5000)
