from model import Model
from flask import Flask, request, jsonify
import os

app = Flask(__name__)

# This should be changed to fit server's address
# Current path "http_://127.0.0.1:5000/face-recognition/authenticate"
@app.route('/face-recognition/authenticate', methods=['POST'])
def authenticate():
    # Path might need to be changed
    model_path = "Project-Face-ID/models/model.pt"
    temp_image_name = "temp_img.png"

    #model = Model()
    #model.load(model_path)

    if 'image' not in request.files:
        return jsonify({"error": "No image part in the request"}), 400

    img = request.files['image']

    if img.filename == '':
        return jsonify({"error": "No image path found"}), 400
    
    if 'user' not in request.form:
        return jsonify({"error": "No user part in the request"}), 400
    
    user = request.form['user']
    
    img.save(temp_image_name)
    abs_img_path = os.path.abspath(temp_image_name)

    #result = model.eval(abs_img_path)
    os.remove(abs_img_path)

    #if result == user:
    #    return jsonify({"result": 1})
    #else:
    #    return jsonify({"result": 0})

    return jsonify({"result": "susy bakka"})

if __name__ == '__main__':
    # Add debug=True for debugging
    app.run(debug=True)
