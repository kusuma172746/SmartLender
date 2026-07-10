from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load Model and Scaler
model = pickle.load(open("Training/rdf.pkl", "rb"))
scaler = pickle.load(open("Training/scale.pkl", "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():

    if request.method == "GET":
        return render_template("input.html")

    # -------- Get Form Data --------

    gender = request.form["Gender"]
    married = request.form["Married"]
    dependents = request.form["Dependents"]
    education = request.form["Education"]
    self_employed = request.form["Self_Employed"]

    applicant_income = float(request.form["ApplicantIncome"])
    coapplicant_income = float(request.form["CoapplicantIncome"])
    loan_amount = float(request.form["LoanAmount"])
    loan_term = float(request.form["Loan_Amount_Term"])

    credit_history = request.form["Credit_History"]
    property_area = request.form["Property_Area"]

    # -------- Convert Text to Numbers --------

    gender = 1 if gender == "Male" else 0

    married = 1 if married == "Yes" else 0

    education = 0 if education == "Graduate" else 1

    self_employed = 1 if self_employed == "Yes" else 0

    credit_history = 1 if credit_history == "Good" else 0
        # Convert Dependents
    if dependents == "0":
        dependents = 0
    elif dependents == "1":
        dependents = 1
    elif dependents == "2":
        dependents = 2
    else:
        dependents = 3

    # Convert Property Area
    if property_area == "Rural":
        property_area = 0
    elif property_area == "Semiurban":
        property_area = 1
    else:
        property_area = 2

    # Prepare Input Data
    features = np.array([[
        gender,
        married,
        dependents,
        education,
        self_employed,
        applicant_income,
        coapplicant_income,
        loan_amount,
        loan_term,
        credit_history,
        property_area
    ]])

    # Scale Input
    features = scaler.transform(features)

    # Make Prediction
    prediction = model.predict(features)

    if prediction[0] == 1:
        result = "🎉 Loan Approved"
    else:
        result = "❌ Loan Rejected"

    return render_template("output.html", prediction=result)


if __name__ == "__main__":
    app.run(debug=True)