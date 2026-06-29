from flask import Flask, render_template, request
import fitz
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

UPLOAD_FOLDER = "resumes"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

def extract_text(pdf_path):
    text = ""
    doc = fitz.open(pdf_path)

    for page in doc:
        text += page.get_text()

    return text


@app.route("/", methods=["GET", "POST"])
def index():

    results = []

    if request.method == "POST":

        job_description = request.form["job"]

        files = request.files.getlist("resumes")

        os.makedirs(UPLOAD_FOLDER, exist_ok=True)

        for file in files:

            if file.filename == "":
                continue

            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            resume_text = extract_text(filepath)

            vectorizer = TfidfVectorizer()

            vectors = vectorizer.fit_transform(
                [job_description, resume_text]
            )

            score = cosine_similarity(vectors[0], vectors[1])[0][0]

            results.append(
                (
                    file.filename,
                    round(score * 100, 2)
                )
            )

        results.sort(key=lambda x: x[1], reverse=True)

    return render_template("index.html", results=results)


if __name__ == "__main__":
    app.run(debug=True)