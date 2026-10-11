# Heart Disease Prediction System

**Live demo (Streamlit):** _https://codingwith-hamza-heart-disease-predictor-app-n53wqy.streamlit.app/_

An end-to-end Machine Learning system that predicts whether a patient has heart
disease from **4 clinical attributes**, built for **AIC354 Machine Learning
Fundamentals — Assignment 2 (Task 2)**.

This project follows the exact same nine-step pipeline as the instructor's
Titanic example: _Import → Load → Understand/Pre-process → Feature Extraction →
Label Encoding → Training → Testing → Application → Feedback_, plus cloud
deployment.

## Input attributes (only 4, as required)

| Attribute        | Meaning                     | Values                                                       |
| ---------------- | --------------------------- | ------------------------------------------------------------ |
| **Age**          | Patient age (years)         | 29–77                                                        |
| **Sex**          | Biological sex              | Male, Female                                                 |
| **ChestPain**    | Type of chest pain          | Typical Angina, Atypical Angina, Non-Anginal Pain, Asymptomatic |
| **MaxHeartRate** | Maximum heart rate achieved | 71–202 bpm                                                   |

**Output:** `HeartDisease` → Yes / No

## Dataset

- **Source:** UCI Machine Learning Repository — Heart Disease (dataset id 45, Cleveland subset)
  <https://archive.ics.uci.edu/dataset/45/heart+disease>
- `heart_disease.csv` — 303 records.

## Files

| File | Description |
| ---- | ----------- |
| `Heart_Disease_Prediction.ipynb` | The full notebook (all 9 steps + deployment section) |
| `heart_disease.csv` | The dataset |
| `svc_trained_model.pkl` | The trained Support Vector Classifier |
| `scaler.pkl` | The fitted `StandardScaler` used before the model |
| `app.py` | Streamlit web app for the live demo |
| `requirements.txt`, `runtime.txt` | Deployment dependencies (Python 3.11) |

## Results

- Algorithm: **Support Vector Classifier** (`SVC(gamma='auto')`) on standardized features
- Test accuracy: **~0.79** with balanced precision/recall between the two classes

## Run the notebook locally

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook                # open Heart_Disease_Prediction.ipynb, Run All
```

## Run the web app locally

```bash
pip install streamlit
streamlit run app.py
```

## Deploy on Streamlit Community Cloud

1. Create a **public** GitHub repo and upload `app.py`, `svc_trained_model.pkl`,
   `scaler.pkl`, `requirements.txt`, and `runtime.txt` to the repo root.
2. Go to <https://share.streamlit.io> → **Create app** → deploy from GitHub.
3. Pick the repo, branch `main`, main file `app.py`, and set **Python version = 3.11**.
4. Copy the public `.streamlit.app` URL to the top of the notebook and this README.
