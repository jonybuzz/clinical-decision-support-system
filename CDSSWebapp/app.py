from flask import Flask, request, jsonify, session
from flask_cors import CORS  # Import CORS
from logic.file_logic import save_file, allowed_file
import csv
from flask import send_from_directory
import os

app = Flask(__name__)
CORS(app, supports_credentials=True)
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
        session['username'] = username  # Store username in session
        # If validation is successful
        return jsonify({"message": "Login successful"}), 200
    else:
        # If validation fails
        return jsonify({"error": "Invalid credentials"}), 401


@app.route('/logout')
def logout():
    session.pop('username', None)  # Remove username from session
    return jsonify({"message": "You have been logged out"}), 200


@app.route('/user')
def session_data():
    if 'username' in session:
        # User is logged in, serve session data
        return jsonify({"username": session['username']})
    else:
        # User is not logged in, return unauthorized error
        return jsonify({"error": "Unauthorized"}), 401


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401  # Ensure the user is logged in
    username = session['username']
    success, filename = save_file(file, app.config['UPLOAD_FOLDER'], username)
    if success:
        return jsonify({"message": f"File {filename} uploaded successfully"}), 200
    else:
        return jsonify({"error": "File not allowed"}), 400


@app.route('/files/<filename>')
def serve_file_content(filename):
    # Ensure the file exists and is a CSV file
    if not allowed_file(filename):
        return jsonify({"error": "File not allowed"}), 400

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    # Read the CSV file and convert its content into a list of dictionaries
    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        file_content = [row for row in csv_reader]

    # Return the file content as JSON
    return jsonify(file_content)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
