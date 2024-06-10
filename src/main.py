from model import Model
from flask import Flask, request, jsonify
from augment import augment
import os
from moviepy.editor import VideoFileClip
from PIL import Image

app = Flask(__name__)

# This should be changed to fit server's address
# Current path "http_://127.0.0.1:5000/face-recognition/authenticate"
@app.route('/face-recognition/authenticate', methods=['POST'])
def authenticate():

    PROJECT_ROOT = os.getenv('PROJECT_ROOT', os.path.dirname(__file__))
    if 'user' not in request.form:
        return jsonify({"error": "No user part in the request"}), 400
    user = request.form['user']

    #model_path = os.path.join("/models", f"{user}.pt")
    #model_path = f"/models/{user}.pt"
    model_path = os.path.join(PROJECT_ROOT, "models", f"{user}.pt")
    model = Model(classes=2)

    if 'purpose' not in request.form:
        return jsonify({"error": "No purpose part in the request"}), 400
    purpose = request.form['purpose']

    if purpose == "train":
        if os.path.exists(model_path):
            return jsonify({"error": "Model for this user already exists, cannot train a new one"}), 400
        else:
            if 'video' not in request.files:
                return jsonify({"error": "No video part in the request"}), 400
            vid = request.files['video']

            if vid.filename == '':
                return jsonify({"error": "No image path found"}), 400

            temp_video_name = "temp_vid.mp4"
            vid.save(temp_video_name)
            abs_vid_path = os.path.abspath(temp_video_name)

            clip = VideoFileClip(abs_vid_path)
            frame_count = 0
            training_images_path = os.path.join("Project-Face-ID", "trainingImages")

            if not os.path.exists(training_images_path):
                os.makedirs(training_images_path)

            for frame in clip.iter_frames():
                frame_filename = os.path.join(training_images_path, f"frame_{frame_count:04d}.png")
                frame_image = Image.fromarray(frame)
                frame_image.save(frame_filename)
                frame_count += 1

            clip.close()
            os.remove(abs_vid_path)

            augment(training_images_path, training_images_path, 1000)

            model.train(training_images_path, 10, 32, 0.001)
            model.save(model_path)

            # Delete all training images
            for file in os.listdir(training_images_path):
                os.remove(os.path.join(training_images_path, file))
            

            return jsonify({"result": 1})
            # return jsonify({"error": "This feature is not yet implemented"}), 400
    elif purpose == "auth":
        if not os.path.exists(model_path):
            return jsonify({"error": "There is no model trained for this user"}), 400
        else:
            model.load(model_path)

            if 'image' not in request.files:
                return jsonify({"error": "No image part in the request"}), 400
            img = request.files['image']

            if img.filename == '':
                return jsonify({"error": "No image path found"}), 400

            temp_image_name = "temp_img.png"
            img.save(temp_image_name)
            #abs_img_path = os.path.abspath(temp_image_name)

            choice, threshold, probs = model.eval(temp_image_name)

            for i, p in enumerate(probs):
                if i == choice:
                    if p >= threshold:
                        return jsonify({"result": 1}) # Image class was found

            os.remove(temp_image_name)
            return jsonify({"result": 0}) # Image class was not found
    else:
        return jsonify({"error": "The purpose is invalid"}), 400

if __name__ == '__main__':
    # Add debug=True for debugging
    app.run(debug=True)
