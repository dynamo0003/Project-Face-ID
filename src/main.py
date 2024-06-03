from model import Model
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# This should be changed to fit server's address
# Current path "http_://127.0.0.1:5000/face-recognition/authenticate"
@app.route('/face-recognition/authenticate', methods=['POST'])
def authenticate():
    # Replace "temp_path" with actual path once connected to server
    model_path = "temp_path"
    temp_image_name = "temp_img.png"

    model = Model()
    model.load(model_path)

    img = request.files['image']
    img.save(temp_image_name)
    abs_img_path = os.path.abspath(temp_image_name)

    model.eval(abs_img_path)

    os.remove(abs_img_path)

if __name__ == '__main__':
    pass
