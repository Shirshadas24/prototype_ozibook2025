# --- BACKEND: backend_app.py ---
from flask import Flask, request, jsonify
from flask_cors import CORS

import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer, LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
import shap
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.utils import resample
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

app = Flask(__name__)
CORS(app)

# Setup paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, '../data/sample_project_data_enhanced.csv')

# Load data
df = pd.read_csv(DATA_PATH)
df['tech_stack'] = df['tech_stack'].apply(lambda x: x.split(';'))

# Encode categorical features
mlb = MultiLabelBinarizer()
tech_encoded = mlb.fit_transform(df['tech_stack'])
tech_df = pd.DataFrame(tech_encoded, columns=mlb.classes_)

le_domain = LabelEncoder()
df['domain_encoded'] = le_domain.fit_transform(df['project_domain'])

le_complexity = LabelEncoder()
df['complexity_encoded'] = le_complexity.fit_transform(df['project_complexity'])

le_urgency = LabelEncoder()
df['urgency_encoded'] = le_urgency.fit_transform(df['deadline_urgency'])

le_team = LabelEncoder()
df['team_encoded'] = le_team.fit_transform(df['assigned_team'])

scaler_delivery = MinMaxScaler()
df[['delivery_time']] = scaler_delivery.fit_transform(df[['delivery_time']])

scaler_project = MinMaxScaler()
df[['project_size']] = scaler_project.fit_transform(df[['project_size']])



balanced_df = pd.DataFrame()
for team in df['assigned_team'].unique():
    group = df[df['assigned_team'] == team]
    if len(group) < 100:
        group = resample(group, replace=True, n_samples=100, random_state=42)
    balanced_df = pd.concat([balanced_df, group])

df = balanced_df.reset_index(drop=True)

df['interaction_1'] = df['delivery_time'] * df['complexity_encoded']
df['interaction_2'] = df['client_rating'] / (df['urgency_encoded'] + 1)
df['interaction_3'] = df['project_size'] * df['team_performance']

# Combine features
X = pd.concat([
    df[['domain_encoded', 'delivery_time', 'complexity_encoded', 'client_rating',
        'project_size', 'urgency_encoded', 'team_performance', 'team_workload','interaction_1', 'interaction_2', 'interaction_3']],
    tech_df
], axis=1)

y = df['team_encoded']


imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_imputed, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, stratify=y_resampled, random_state=42
)

model = XGBClassifier(eval_metric='mlogloss', learning_rate= 0.3, max_depth= 5, n_estimators= 150, subsample= 0.8)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# Compute evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

# Print or log them (or write to a file if needed)
print(df.shape)
print("\n Model Evaluation Metrics:")
print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")

explainer = shap.TreeExplainer(model)


# Save encoders and model
joblib.dump(model, os.path.join(BASE_DIR, 'model.pkl'))
joblib.dump(mlb, os.path.join(BASE_DIR, 'mlb.pkl'))
joblib.dump(le_domain, os.path.join(BASE_DIR, 'le_domain.pkl'))
joblib.dump(le_complexity, os.path.join(BASE_DIR, 'le_complexity.pkl'))
joblib.dump(le_urgency, os.path.join(BASE_DIR, 'le_urgency.pkl'))
joblib.dump(le_team, os.path.join(BASE_DIR, 'le_team.pkl'))
joblib.dump(scaler_project, os.path.join(BASE_DIR, 'scaler_project.pkl'))
joblib.dump(scaler_delivery, os.path.join(BASE_DIR, 'scaler_delivery.pkl'))


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    try:
        domain = data.get('project_domain')
        tech_stack = data.get('tech_stack', [])
        delivery_time = int(data.get('delivery_time'))
        complexity = data.get('project_complexity')
        rating = float(data.get('client_rating'))
        size = int(data.get('project_size'))
        urgency = data.get('deadline_urgency')
        perf = float(data.get('team_performance'))
        workload = int(data.get('team_workload'))

        interaction1 = rating * perf
        interaction2 = size * delivery_time
        interaction3 = rating * workload

        # Load encoders/models
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model = joblib.load(os.path.join(BASE_DIR, 'model.pkl'))
        mlb = joblib.load(os.path.join(BASE_DIR, 'mlb.pkl'))
        le_domain = joblib.load(os.path.join(BASE_DIR, 'le_domain.pkl'))
        le_complexity = joblib.load(os.path.join(BASE_DIR, 'le_complexity.pkl'))
        le_urgency = joblib.load(os.path.join(BASE_DIR, 'le_urgency.pkl'))
        le_team = joblib.load(os.path.join(BASE_DIR, 'le_team.pkl'))
        scaler_project = joblib.load(os.path.join(BASE_DIR, 'scaler_project.pkl'))
        scaler_delivery = joblib.load(os.path.join(BASE_DIR, 'scaler_delivery.pkl'))

        # Encode features
        tech_vector = [1 if tech in tech_stack else 0 for tech in mlb.classes_]
        domain_encoded = le_domain.transform([domain])[0]
        complexity_encoded = le_complexity.transform([complexity])[0]
        urgency_encoded = le_urgency.transform([urgency])[0]

        # Normalize delivery time and project size
        delivery_scaled = scaler_delivery.transform([[delivery_time]])[0][0]
        size_scaled = scaler_project.transform([[size]])[0][0]

        input_features = [domain_encoded, delivery_scaled, complexity_encoded, rating,
                          size_scaled, urgency_encoded, perf, workload] +[interaction1, interaction2, interaction3]+ tech_vector

        # Prediction
        proba = model.predict_proba([input_features])[0]
        top2_idx = proba.argsort()[-2:][::-1]

        top_team = le_team.inverse_transform([top2_idx[0]])[0]
        second_team = le_team.inverse_transform([top2_idx[1]])[0]


        input_array = np.array([input_features])

        # SHAP explanation
        shap_vals_all = explainer.shap_values(input_array)

        if isinstance(shap_vals_all, list):
            shap_vals = shap_vals_all[top2_idx[0]][0]
        else:
            shap_vals = shap_vals_all[0]

        shap_vals = shap_vals.flatten()

        top_explanations = sorted(zip(X.columns, shap_vals), key=lambda x: abs(x[1]), reverse=True)[:3]
        explanation = [{'feature': str(f), 'impact': round(float(i), 3)} for f, i in top_explanations]

        
        summary_reasons = []

# Include SHAP-important tech skills from user input
        for feature, impact in top_explanations:
            if feature in mlb.classes_ and feature in tech_stack:
                summary_reasons.append(f"{feature} skill")

# Add human-readable explanations for important project inputs
        # Add project complexity
        if complexity == "low":
            summary_reasons.append("low complexity")
        elif complexity == "high":
            summary_reasons.append("high complexity")

# Add urgency
        if urgency == "high":
            summary_reasons.append("high urgency")

# Add client rating
        if rating >= 4.1:
            summary_reasons.append("high client rating")
        elif rating <= 3.5:
            summary_reasons.append("low client rating")

# Add delivery time
        if delivery_time <= 30:
            summary_reasons.append("short delivery time")
        elif delivery_time >= 180:
            summary_reasons.append("long delivery time")

#

#   Add project size buckets
        if size >= 800:
            summary_reasons.append("large project size")
        elif size <= 300:
            summary_reasons.append("small project size")

        summary_reasons.append(f"{ ', '.join(tech_stack)} skill(s) and {domain} domain")

# Join into a sentence
        if summary_reasons:
            summary_text = "The first team was selected due to " + ", ".join(summary_reasons) + "."
        else:
            summary_text = "The team was selected based on important contributing features."




        response = {
    "recommended_team": str(top_team),
            "confidence": round(float(proba[top2_idx[0]]) * 100, 2),
            "alternate_team": str(second_team),
            "alternate_confidence": round(float(proba[top2_idx[1]]) * 100, 2),
        
    "explanation": explanation,
        "summary": summary_text

}


        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)
