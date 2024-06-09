import platform
from model import Model
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# This should be changed to fit server's address
# Current path "http_://127.0.0.1:5000/face-recognition/authenticate"
@app.route('/face-recognition/authenticate', methods=['POST'])
def authenticate():
    if 'user' not in request.form:
        return jsonify({"error": "No user part in the request"}), 400
    user = request.form['user']

    model_path = f"Project-Face-ID/models/{user}.pt"
    if not os.path.exists(model_path):
        # return jsonify({"error": f"Model file {model_path} does not exist"}), 400
        # Code to train new model
        pass
    else:
        classes = 4
        model = Model(classes)
        model.load(model_path)

        if 'image' not in request.files:
            return jsonify({"error": "No image part in the request"}), 400
        img = request.files['image']

        if img.filename == '':
            return jsonify({"error": "No image path found"}), 400

        temp_image_name = "temp_img.png"
        img.save(temp_image_name)
        abs_img_path = os.path.abspath(temp_image_name)

        choice, threshold, probs = model.eval(abs_img_path)
        os.remove(abs_img_path)

        for i, p in enumerate(probs):
            if i == choice:
                if p >= threshold:
                    return jsonify({"result": 1}) # Image class was found
        
        return jsonify({"result": 0}) # Image class was not found

if __name__ == '__main__':
    # Add debug=True for debugging
    app.run(debug=True)
