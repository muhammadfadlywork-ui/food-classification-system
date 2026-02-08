import os

class FileHandler:
    def __init__(self, upload_folder):
        self.upload_folder = upload_folder

    def save_file(self, file):
        filename = file.filename
        filepath = os.path.join(self.upload_folder, filename)
        file.save(filepath)
        return filename, filepath
