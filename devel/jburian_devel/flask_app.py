from flask import Flask
from flask import request, jsonify, render_template
import sys

import loguru
from loguru import logger

from pathlib import Path

import os


# Upload file


# Trénování
# curl -X GET 147.228.140.130:5000/train?filenames="img1.czi,img2.czi"?modelname="mujmodel"

# predikce s modelem
# curl -X GET 147.228.140.130:5000/predict?filename="img100.czi"?modelname="mujmodel"

UPLOAD_FOLDER = Path('C:\Temp')
ALLOWED_EXTENSIONS = {'czi'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



def export_czi_to_jpg():
    pass


def create_COCO_json():
    pass


def create_COCO():
    pass


@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/exists", methods=["GET", "POST"])
def exists():
    if request.method == "POST":
        filename = request.args.get("filename")
        #print(filename)
        file_existence = Path(filename).exists()
        logger.debug(f"file_exists={file_existence}")
        return jsonify(file_existence)
        # return jsonify({"file_exists": file_exists})
    return jsonify({})


@app.route("/upload", methods=["GET", "POST"])
def upload():
    if request.method == "POST":
        filename = request.args.get("filename")

        if not Path(filename).exists():
            logger.debug(f"File does not exist. filename={filename}")
            return jsonify({"error": "File does not exists."})

        if not (allowed_file(filename)):
            logger.debug(f"Uploaded file is in bad format. filename={filename}")
            return jsonify({"error": "Uploaded file is in bad format."})


@app.route("/train", methods=["GET", "POST"])
def train():
    # filename = request.args.get("filename")
    print(request.args, file=sys.stderr)
    return request.args["filenames"]


@app.route("/predict", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        filename = request.args.get("filename")

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
