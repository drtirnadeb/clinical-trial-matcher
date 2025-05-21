import gradio as gr
import pandas as pd
import numpy as np
import joblib

# Load the model and trial data
model = joblib.load("trial_match_model_numeric.pkl")
trials = pd.read_csv("numeric_cleaned_trials.csv")

def recommend_ml_numeric(age, weight, height, bio_sex, smoking, medication, pcos, menstruation):
    bmi = (weight / (height ** 2)) * 703
    results = []

    for _, trial in trials.iterrows():
        try:
            features = {
                'Age': age,
                'BMI': bmi,
                'Biological_Sex': bio_sex,
                'Smoking_Status': smoking,
                'On_Medication': medication,
                'PCOS': pcos,
                'Regular_Menstruation': menstruation,
                'Trial_Sex': trial['Biological Sex'],
                'Trial_Smoking': trial['Smoking status'],
                'Trial_Medication': trial['Current Medication'],
                'Trial_PCOS': trial['PCOS'],
                'Trial_Menstruation': trial['Regular menstruation?']
            }

            X = pd.DataFrame([features])
            score = model.predict_proba(X)[0][1]

            matched = []
            if trial['Age_Min'] <= age <= trial['Age_Max']:
                matched.append("✅ Age")
            if trial['BMI_Min'] <= bmi <= trial['BMI_Max']:
                matched.append("✅ BMI")
            if trial['Biological Sex'] == bio_sex:
                matched.append("✅ Sex")
            if trial['Smoking status'] == smoking:
                matched.append("✅ Smoking")
            if trial['Current Medication'] == medication:
                matched.append("✅ Meds")
            if trial['PCOS'] == pcos:
                matched.append("✅ PCOS")
            if trial['Regular menstruation?'] == menstruation:
                matched.append("✅ Menstruation")

            explanation = "; ".join(matched)
            results.append((trial['Clinical Trial'], trial['urls'], score, explanation))

        except Exception as e:
            continue

    top_matches = sorted(results, key=lambda x: x[2], reverse=True)[:3]

    if top_matches:
        return "\n\n".join([
            f"🔗 **[{title}]({url})**\nMatch Score: {score:.2f}\n**Why matched:** {explanation}"
            for title, url, score, explanation in top_matches
        ])
    else:
        return "😕 No strong match found. Try adjusting your input."

app = gr.Interface(
    fn=recommend_ml_numeric,
    inputs=[
        gr.Number(label="Age"),
        gr.Number(label="Weight (lbs)"),
        gr.Number(label="Height (in)"),
        gr.Radio([0, 1], label="Biological Sex (0=female, 1=male)"),
        gr.Radio([0, 1], label="Smoking Status (0=no, 1=yes)"),
        gr.Radio([0, 1], label="Currently on Medication? (0=no, 1=yes)"),
        gr.Radio([0, 1], label="Diagnosed with PCOS? (0=no, 1=yes)"),
        gr.Radio([0, 1], label="Regular Menstruation? (0=no, 1=yes)")
    ],
    outputs="markdown",
    title="🧬 Clinical Trial Matcher (Numeric)",
    description="Fill in your info (0/1 for booleans) to get matched to top trials using a trained ML model."
)

if __name__ == "__main__":
    app.launch()


