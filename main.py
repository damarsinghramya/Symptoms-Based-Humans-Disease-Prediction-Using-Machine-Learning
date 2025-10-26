


from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and encoders
model = joblib.load('random_forest_model.pkl')
le = joblib.load('label_encoder.pkl')
feature_columns = joblib.load('feature_columns.pkl')

# Load dataset to get unique symptoms
dataset = pd.read_csv('dataset.csv')
symptom_columns = pd.unique(dataset.drop('Disease', axis=1).values.ravel())
symptom_columns = [s.strip() for s in symptom_columns if pd.notnull(s)]

# Load precaution and doctor mapping
precaution_df = pd.read_csv('symptom_precaution.csv')
doctor_df = pd.read_csv('Doctor_Versus_Disease.csv', encoding='latin1')

# Clean column names
precaution_df.columns = precaution_df.columns.str.strip()
doctor_df.columns = doctor_df.columns.str.strip()

# ✅ Welcome route
@app.route('/')
def welcome():
    return render_template('welcome.html')

# ✅ Login route
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if username == 'patient' and password == '123':
            return redirect(url_for('index'))
        else:
            return render_template('login.html', error="Invalid credentials. Try again.")

    return render_template('login.html')

# ✅ Index route
@app.route('/index')
def index():
    return render_template('index.html', symptoms=symptom_columns)

# ✅ Predict route – modified to render result on a new page
@app.route('/predict', methods=['POST'])
def predict():
    selected_symptoms = request.form.getlist('symptoms')

    if not selected_symptoms:
        error = "Please select at least one symptom."
        return render_template('index.html', symptoms=symptom_columns, error=error)

    input_dict = {feature: 0 for feature in model.feature_names_in_}
    model_features_cleaned = [f.strip().lower() for f in model.feature_names_in_]

    for symptom in selected_symptoms:
        clean_symptom = symptom.strip().lower()
        for i, feature in enumerate(model_features_cleaned):
            if clean_symptom == feature:
                actual_feature = model.feature_names_in_[i]
                input_dict[actual_feature] = 1

    input_df = pd.DataFrame([input_dict])

    predicted_encoded = model.predict(input_df)[0]
    predicted_label = le.inverse_transform([predicted_encoded])[0]

    # Precautions
    precaution_row = precaution_df[precaution_df.iloc[:, 0] == predicted_label]
    precautions = precaution_row.iloc[0, 1:].dropna().tolist() if not precaution_row.empty else []

    # Doctor recommendation
    doctor_row = doctor_df[doctor_df.iloc[:, 0] == predicted_label]
    doctor = doctor_row.iloc[0, 1] if not doctor_row.empty else "No doctor found."

    # ✅ Render result.html page with prediction output
    return render_template('result.html',
                           predicted_disease=predicted_label,
                           precautions=precautions,
                           doctor=doctor)

if __name__ == '__main__':
    app.run(debug=True)

