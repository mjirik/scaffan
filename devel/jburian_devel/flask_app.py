from flask import Flask
from flask import request, jsonify, render_template
import sys
import os


# Upload file


# Trénování
# curl -X GET 147.228.140.130:5000/train?filenames="img1.czi,img2.czi"?modelname="mujmodel"

# predikce s modelem
# curl -X GET 147.228.140.130:5000/predict?filename="img100.czi"?modelname="mujmodel"


app = Flask(__name__)

def export_czi_to_jpg():
    pass

def create_COCO_json():
    pass

def create_COCO():
    pass

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"


@app.route("/upload", methods=["GET", "POST"])
def exists():
    if request.method == "POST":
        filename = request.args.get("filename")
        print(filename)
        # exists = Path(filename).exists()
        # logger.debug(f"exists={exists}")
        return jsonify(exists)
        # return jsonify({"exists": exists})
    return jsonify({})


@app.route("/train", methods=["GET", "POST"])
def train():
    # filename = request.args.get("filename")
    print(request.args, file=sys.stderr)
    return request.args["filenames"]


@app.route("/predict", methods=["GET", "POST"])
def predict():
    pass



if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
