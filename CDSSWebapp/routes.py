from flask import Blueprint, request, jsonify, session
from logic.file_logic import save_file, allowed_file
import csv
import os
import config

routes = Blueprint('routes', __name__)


@routes.route('/login', methods=['POST'])
def login():
    username = request.json.get('username')
    password = request.json.get('password')
    users = config.USERS

    if username in users and users[username] == password:
        session['username'] = username
        return jsonify({"message": "Login successful"}), 200
    else:
        return jsonify({"error": "Invalid credentials"}), 401


@routes.route('/logout')
def logout():
    session.pop('username', None)
    return jsonify({"message": "You have been logged out"}), 200


@routes.route('/user')
def session_data():
    if 'username' in session:
        return jsonify({"username": session['username']})
    else:
        return jsonify({"error": "Unauthorized"}), 401


@routes.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files['file']
    if 'username' not in session:
        return jsonify({"error": "Unauthorized"}), 401
    username = session['username']
    success, filename = save_file(file, config.UPLOAD_FOLDER, username)
    if success:
        return jsonify({"message": f"File {filename} uploaded successfully"}), 200
    else:
        return jsonify({"error": "File not allowed"}), 400


@routes.route('/files/<filename>')
def serve_file_content(filename):
    if not allowed_file(filename):
        return jsonify({"error": "File not allowed"}), 400

    file_path = os.path.join(config.UPLOAD_FOLDER, filename)
    if not os.path.exists(file_path):
        return jsonify({"error": "File not found"}), 404

    with open(file_path, mode='r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        file_content = [row for row in csv_reader]

    return jsonify(file_content)
