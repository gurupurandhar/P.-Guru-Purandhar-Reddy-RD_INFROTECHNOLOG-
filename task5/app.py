from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)

model = load_model("mask_detector.h5")

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    prediction = None

    if request.method == "POST":

        file = request.files["image"]

        filepath = os.path.join(UPLOAD_FOLDER, file.filename)

        file.save(filepath)

        img = image.load_img(filepath, target_size=(128,128))

        img = image.img_to_array(img)

        img = np.expand_dims(img, axis=0)

        img = img/255.0

        pred = model.predict(img)

        if pred[0][0] > 0.5:

            prediction = "Without Mask"

        else:

            prediction = "With Mask"

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run(debug=True)