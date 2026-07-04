from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")


@app.route("/", methods=["GET", "POST"])
def index():

    prediction = ""

    if request.method == "POST":

        text = request.form["text"]

        text_vector = vectorizer.transform([text])

        result = model.predict(text_vector)

        prediction = result[0]

    return render_template("index.html", prediction=prediction)


if __name__ == "__main__":
    app.run(debug=True)