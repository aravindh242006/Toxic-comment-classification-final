
from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("toxic_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    comment = request.form.get("comment", "")
    transformed = vectorizer.transform([comment])

    pred = model.predict(transformed)[0]

   
    proba = model.predict_proba(transformed)[0]  

    toxic_confidence = proba[1] * 100
    clean_confidence = proba[0] * 100      

    if pred == 1:
        result = f"Toxic Comment ({toxic_confidence:.2f}% confidence)"
    else:
        result = f"Clean Comment ({clean_confidence:.2f}% confidence)"

    return render_template("result.html", comment=comment, result=result)

if __name__ == "__main__":
    app.run(debug=True)
