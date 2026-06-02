
from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    lot_area = float(request.form["lot_area"])
    yr_sold = float(request.form["yr_sold"])

    prediction = model.predict(np.array([[lot_area, yr_sold]]))[0]

    return render_template(
        "index.html",
        prediction=f"Predicted Sale Price: ${prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
