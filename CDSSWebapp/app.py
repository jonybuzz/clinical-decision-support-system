from flask import Flask, request, jsonify, session
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes and domains
app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

users = {'admin': '123'}


@app.route('/login', methods=['POST'])
def login():
    # Extract username and password from request data
    username = request.json.get('username')
    password = request.json.get('password')

    # Validate username and password
    if username in users and users[username] == password:
        # If validation is successful
        return jsonify({"message": "Login successful"}), 200
    else:
        # If validation fails
        return jsonify({"error": "Invalid credentials"}), 401


@app.route('/upload', methods=['POST'])
def upload_file():
    # Your existing file upload logic
    pass


if __name__ == '__main__':
    app.run(debug=True, port=5000)
