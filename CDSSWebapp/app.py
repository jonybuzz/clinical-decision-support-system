from flask import Flask, request, jsonify, session
from flask_cors import CORS  # Import CORS
import os
import secrets
import config
from routes import routes  # Import the routes blueprint

app = Flask(__name__)
CORS(app, supports_credentials=True)
app.secret_key = secrets.token_hex(32)  # Generate a secure random string for signing cookies
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = config.UPLOAD_FOLDER
app.config['DEBUG'] = config.DEBUG
app.config['TESTING'] = config.TESTING
app.config['SESSION_COOKIE_NAME'] = config.SESSION_COOKIE_NAME
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH

os.makedirs(config.UPLOAD_FOLDER, exist_ok=True)

users = config.USERS

app.register_blueprint(routes)  # Register the blueprint


if __name__ == '__main__':
    app.run(debug=app.config['DEBUG'], port=config.PORT)
