from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

model = joblib.load("fake_news_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = ""

    if request.method == "POST":

        news = request.form["news"]

        news_vector = vectorizer.transform([news])

        result = model.predict(news_vector)

        if result[0] == 1:
            prediction = "✅ Real News"
        else:
            prediction = "❌ Fake News"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)