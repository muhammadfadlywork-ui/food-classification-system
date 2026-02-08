from flask import Flask, render_template, request, redirect, url_for
from classifier import ImageClassifier
from file_handler import FileHandler
import os

class FlaskApp:
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['UPLOAD_FOLDER'] = 'static/uploads'


        self.classifier = ImageClassifier(
            "static/model/V2S_BestModel.pth",     # <-- ganti ke .pth kamu
            "static/model/classes.json",          # <-- saranku pakai ini biar urutan class aman
            target_size=(380, 380)                # <-- 380 sesuai training yang kamu minta
        )

        self.file_handler = FileHandler(self.app.config['UPLOAD_FOLDER'])
        self.setup_routes()

    def setup_routes(self):
        @self.app.route('/')
        def index():
            return render_template('belajar.html')

        @self.app.route('/predict', methods=['GET', 'POST'])
        def predict():
            if request.method == 'POST':
                file = request.files.get('file')
                if not file:
                    return "Tidak ada file yang diupload"

                filename, filepath = self.file_handler.save_file(file)
                pred_class = self.classifier.predict(filepath)
                return redirect(url_for('result', img=filename, pred=pred_class))

            return render_template('predict.html')

        @self.app.route('/result')
        def result():
            img_file = request.args.get('img')
            pred_class = request.args.get('pred')

            if not img_file or not pred_class:
                return "Tidak ada hasil yang ditampilkan"

            img_path = os.path.join(self.app.config['UPLOAD_FOLDER'], img_file)
            return render_template('result.html', image_path=img_path, result=pred_class)

    def run(self):
        self.app.run(debug=True)

if __name__ == '__main__':
    app_instance = FlaskApp()
    app_instance.run()
