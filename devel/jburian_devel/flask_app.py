from flask import Flask
from flask import request, jsonify, render_template
from flask import redirect, url_for
from flask import send_from_directory
from flask import session
import export_czi_json

import sys
from werkzeug.utils import secure_filename

from loguru import logger

from pathlib import Path

import skimage.io
import os

path_to_script = Path("~/projects/scaffan/").expanduser()
sys.path.insert(0, str(path_to_script))
import scaffan.image

import os

# Upload file


# Trénování
# curl -X GET 147.228.140.130:5000/train?filenames="img1.czi,img2.czi"?modelname="mujmodel"

# predikce s modelem
# curl -X GET 147.228.140.130:5000/predict?filename="img100.czi"?modelname="mujmodel"

UPLOAD_FOLDER = "/Temp/Uploaded_files"
ALLOWED_EXTENSIONS = {".czi"}  # povolene formaty

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
# app.config["MAX_CONTENT_LENGTH"] = 16 * 1000 * 1000 # maximalni velikost souboru


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"


# Set the secret key to some random bytes. Keep this really secret!
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/'


@app.route("/")
def index():
    if "username" in session:
        return f'Logged in as {session["username"]}'
    return "You are not logged in"


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        session["username"] = request.form["username"]
        return redirect(url_for("index"))
    return """
        <form method="post">
            <p><input type=text name=username>
            <p><input type=submit value=Login>
        </form>
    """


@app.route("/logout")
def logout():
    # remove the username from the session if it's there
    session.pop("username", None)
    return redirect(url_for("index"))


def export_czi_to_jpg(czi_input_path, jpg_output_path, annotation_name):
    """
    Exports czi files to jpg
    @param czi_input_path: Path path to czi files
    @param jpg_output_path: Path output path to jpg files
    @param annotation_name: Name of saved annotations
    """
    index = 0
    while True:
        fn_str = (
            str(czi_input_path) + "\\" + annotation_name + str(index).zfill(4) + ".czi"
        )
        if not czi_input_path.exists():
            break
        print(f"filename: {czi_input_path} {czi_input_path.exists()}")

        anim = scaffan.image.AnnotatedImage(path=fn_str)

        view = anim.get_full_view(
            pixelsize_mm=[0.0003, 0.0003]
        )  # wanted pixelsize in mm in view
        img = view.get_raster_image()
        os.makedirs(jpg_output_path, exist_ok=True)
        skimage.io.imsave(
            os.path.join(jpg_output_path, str(index).zfill(4) + ".jpg"), img
        )
        index += 1


def create_COCO_json(czi_input_path, jpg_input_path, txt_input_path, json_output_path):
    """

    @param czi_input_path:
    @param jpg_input_path:
    @param txt_input_path:
    @param json_output_path:
    """
    # path jpg input directory
    # path czi input directory
    # path to txt category file
    # path output json file

    pass


def create_COCO(jpg_input_path, json_input_path):
    # completing json file with jpg images and creating COCO dataset
    # saves COCO dataset to certain directory
    # validation, training
    pass


@app.route("/exists", methods=["GET", "POST"])
def exists():
    if request.method == "POST":
        filename = request.args.get("filename")
        # print(filename)
        file_existence = Path(filename).exists()
        logger.debug(f"file_exists={file_existence}")
        return jsonify(file_existence)
        # return jsonify({"file_exists": file_exists})
    return jsonify({})


@app.route("/upload", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":

        # check if the post request has the file part
        if "file" not in request.files:
            logger.debug(f"No file part")
            return jsonify({"No file part"})

        file = request.files["file"]

        test = request.args.get("filename")

        if file.filename == "":
            logger.debug(f"No file selected.")
            return jsonify({"error": "No file selected."})

        if not Path(file.filename).exists():
            logger.debug(f"File does not exist. filename={file.filename}")
            return jsonify({"error": "File does not exist."})

        if not allowed_file(file.filename):
            logger.debug(f"Bad format of the file. filename={file.filename}")
            return jsonify({"error": "Bad format of the file."})

        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config["UPLOAD_FOLDER"], filename))
            logger.debug(f"Everything ok. filename={filename}")
            # return redirect(url_for('download_file', name=filename))
            return jsonify({"OK-200"})
    return """
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    """


@app.route("/uploads/<name>")
def download_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)


@app.route("/train", methods=["GET", "POST"])
def train():
    # filename = request.args.get("filename")
    print(request.args, file=sys.stderr)
    return request.args["filenames"]


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        file = request.files["file"]

    # nacteni obrazku
    # spusteni detectronu s modelem


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
