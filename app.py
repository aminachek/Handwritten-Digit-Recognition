from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np
import os

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model("mnist_cnn_model.h5")


def prepare_image(path):

    img = Image.open(path).convert("L")

    img = img.resize((28,28))

    img = np.array(img).astype("float32")/255.0

    if img.mean() > 0.5:
        img = 1-img

    img = img.reshape(1,28,28,1)

    return img


@app.route("/", methods=["GET","POST"])
def index():

    prediction = None
    confidence = None
    image = None

    if request.method == "POST":

        file = request.files["image"]

        if file:

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)

            file.save(filepath)

            img = prepare_image(filepath)

            pred = model.predict(img)

            prediction = np.argmax(pred)

            confidence = round(np.max(pred)*100,2)

            image = filepath

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           image=image)


if __name__ == "__main__":
    app.run(debug=True)