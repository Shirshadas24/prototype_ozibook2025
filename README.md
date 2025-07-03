# 🤖 Smart Project Assignment System

A full-stack AI-powered web application that recommends the most suitable software development team for a project based on domain, tech stack, timeline, complexity, urgency, and team performance.

---

##  What It Does

This system predicts the **best-fit team** for a project using machine learning and explains its reasoning using SHAP. It offers:

-  Suggested team with confidence
-  Alternate team option
-  SHAP-based feature impact explanation
-  Natural language summary of reasoning
 > The team was selected due to low complexity, high urgency, .NET skill, and finance domain.


---

##  Machine Learning Model

- **Model**: [XGBoost Classifier](https://xgboost.readthedocs.io/en/stable/)
- **Hyperparameters**:
  - `n_estimators`: 150  
  - `max_depth`: 5  
  - `learning_rate`: 0.3  
  - `subsample`: 0.8  
- **Data Balancing**: SMOTE (Synthetic Minority Over-sampling)
- **Multi-class Strategy**: Softmax for multi-team classification


---


## 📊 Performance Metrics

- **Accuracy**: 69%
- **F1 Score**: 69%
- **Precision**: 69%
- **Recall**: 69%
---

## ⚙️ Feature Engineering

-  MultiLabelBinarizer: For multi-label `tech_stack`
-  LabelEncoding: For `project_domain`, `complexity`, `urgency`, and `team`
-  MinMaxScaler: Normalization of `delivery_time` and `project_size`
-  Custom Features:
  - Interactions like `team_workload * urgency`
  - Count of matching tech stack items with historical projects

---

##  How It Works

1. User inputs project info on frontend.
2. Data sent to Flask backend via API.
3. Backend:
   - Preprocesses data using saved encoders/scalers.
   - Predicts the team using XGBoost model.
   - Returns prediction, alternate, confidence & SHAP insights.
4. Frontend displays prediction + explanation beautifully.


##  Frontend (React.js)

-  Transparent form and output cards
-  Sliders with upper limits
-  Form fields:
  - Domain, tech stack, delivery time, complexity, urgency, project size, rating, performance, workload

---

##  Tech Stack

| Layer     | Technology                            |
|-----------|----------------------------------------|
| Frontend  | React.js, Tailwind CSS                 |
| Backend   | Python, Flask, Flask-CORS              |
| ML Model  | Scikit-learn, XGBoost                  |
| Explain   | SHAP                                   |


---
![image](https://github.com/user-attachments/assets/619a1e69-7a02-4f65-88fc-b3457113667c)
![image](https://github.com/user-attachments/assets/af67e9d1-751a-444a-9583-06f35a9519a1)
![image](https://github.com/user-attachments/assets/db41c2c2-90c6-4c05-a5a3-e57506c2dc98)


## ⚠️ Disclaimer

The dataset used in this project was synthetically generated using AI tools and may not fully represent real-world scenarios. As such, model predictions should be interpreted cautiously and are intended primarily for academic or experimental purposes.

## 🔧 Setup Instructions

###  Backend Setup

```bash
cd backend
python -m venv venv
venv\Scripts\activate  # or source venv/bin/activate on Mac/Linux
pip install -r requirements.txt
python backend_app.py
```
###  Frontend Setup
```bash
cd frontend
npm install
npm start
```
> Backend must run at http://localhost:5000. If not, update the proxy URL in frontend/package.json.

---

### 📁 Folder Structure
```java

project-root/
│
├── backend/ # Flask backend + ML models
│ ├── backend_app.py # Main backend script
│ ├── model.pkl # Trained XGBoost model
│ ├── *.pkl # Encoders, scalers, and binarizers
│
├── data/
│ └── sample_project_data_enhanced.csv
│
├── frontend/ # React frontend
│ ├── public/
│   ├── index.html 
│   ├── Icon
│ └── src/
│   ├── App.js # Main UI
│   ├── App.css # Styles
│   ├── index.js 
│   ├── index.css 
│  
├── requirements.txt
├── LICENSE
└── README.md
```
##  Future Scope

- Enhance ML model using advanced architectures like BERT or Graph Neural Networks (GNNs).
- Add support for dynamic real-time project allocation using live team workload APIs.
- Improve SHAP explainability with interactive charts and better frontend visualizations.
- Integrate login & role-based dashboards for project managers and teams.
- Expand dataset with more real-world project scenarios to improve generalization.
---

 
