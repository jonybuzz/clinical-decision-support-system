import os
from werkzeug.utils import secure_filename

ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def save_file(file, upload_folder, username):
    if file and allowed_file(file.filename):
        user_folder = os.path.join(upload_folder, username)
        os.makedirs(user_folder, exist_ok=True)  # Create the user-specific folder if it doesn't exist
        filename = secure_filename(file.filename)
        file.save(os.path.join(user_folder, filename))
        return True, filename  # Indicate success and return the filename
    return False, None  # Indicate failure