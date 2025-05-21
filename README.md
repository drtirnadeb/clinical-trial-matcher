# 🧬 Clinical Trial Matcher for Women’s Health

A machine learning–based prototype designed to help women find clinical trials that match their health profiles. This project was created during the MIT FMML (Female Medicine through Machine Learning) Hackathon by an all–South Asian women team, combining expertise from AI and healthcare.

## 🔍 Background

- Women remain underrepresented in clinical trials.
- PCOS, endometriosis, and other conditions are often under-studied.
- This app connects women to trials using ML-based recommendations.

## 👩‍💻 What This App Does

- Takes a woman’s health profile as input (age, BMI, smoking status, medication, PCOS diagnosis, menstruation regularity).
- Calculates personalized match scores using a trained `RandomForestClassifier`.
- Returns the top matching clinical trials with reasons (age range, sex match, PCOS match, etc).

### 🔗 Try the App (Gradio):
👉 [Live Hugging Face Demo](https://huggingface.co/spaces/tirnadebphd/clinical-trial-matcher)

## 🛠️ How It Works

- Data collected for **8 real PCOS trials in Massachusetts** from clinicaltrials.gov.
- Synthetic patient data created for training.
- Trained a balanced Random Forest classifier on 7 eligibility features.
- Model evaluated with:
  - Accuracy: 96%
  - F1 Score (class 1): 0.85
  - AUC: 0.99
  - Precision@100: 1.0

## 📂 Files

- `Clinical_trail_matcher_notebook.ipynb` — full pipeline: data, model, and Gradio app
- `trial_match_model_numeric.pkl` — trained model (scikit-learn 1.6.1, numpy 2.0)
- `numeric_cleaned_trials.csv` — cleaned trial dataset
- `requirements.txt` — for reproducing environment

## 👥 Team

- **Tirna Deb** — Astrophysicist + Data Scientist (project lead)
- **Melinda Mastan**, **Suci Ardini Widyaningsih**, **Moomtahina Fatima**, **Anusha Senapati** — clinical & data experts

## 🧠 Inspired By

- MIT Female Medicine Through Machine Learning (FMML)
- 130+ attendees from institutions like MIT, Harvard, HMS, Wharton, Johns Hopkins, and more

## 📈 Future Work

- Scrape real-time trials from clinicaltrials.gov
- Use LLMs to extract criteria from trial descriptions
- Scale model beyond PCOS to broader women’s health

---
