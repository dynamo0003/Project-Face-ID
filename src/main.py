from model import Model
from flask import Flask, request, jsonify

app = Flask(__name__)

# This should be changed to fit server's address
# Current path "http_://IP:port/face-recognition/authenticate"
@app.route('/face-recognition/authenticate', methods=['POST'])

def authenticate():
    model_path = "temp_path"
    model = Model()

    model.load(model_path)

if __name__ == '__main__':
    pass
