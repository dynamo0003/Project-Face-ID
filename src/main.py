from model import Model
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/face-recognition/authenticate', methods=['POST'])
def authenticate():
    ...

if __name__ == '__main__':
    pass
