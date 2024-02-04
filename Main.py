import sys
import cv2
import numpy as np
from PyQt5.QtCore import pyqtSignal, QThread
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QFileDialog, QProgressBar, QVBoxLayout, \
    QMessageBox
from PyQt5.QtGui import QPixmap, QFont
from keras.models import load_model
from keras.applications.inception_v3 import preprocess_input
from PyQt5.QtCore import Qt
import os

sys.path.append("./minutaes")
from minutaes import Descriptors
import ShowMinutaeies, ShowMatches


class PredictThread(QThread):
    finished_signal = pyqtSignal(str)

    def __init__(self, image_path, image_width, image_height):
        super().__init__()
        self.predicted_class_label = ""
        self.image_path = image_path
        self.image_width = image_width
        self.image_height = image_height

    def run(self):
        self.predict(self.image_path, self.image_width, self.image_height)
        self.finished_signal.emit(self.predicted_class_label)

    def preProcess(self, image_path, image_width, image_height):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (image_width, image_height))
        img_array = img.reshape(-1, image_width, image_height, 1)  # No need to convert again
        img_array = preprocess_input(img_array)
        print(img.shape)

        return img_array

    def predict(self, image_path, image_width, image_height):
        model = load_model("./models/best model.h5")
        try:
            img_array = self.preProcess(image_path, image_width, image_height)

            predictions = model.predict(img_array)
            print(predictions)
            predicted_class_index = np.argmax(predictions, axis=1)
            print(predicted_class_index)
            classes = ["A", "L", "R", "T", "W"]
            predicted_class_label = classes[predicted_class_index[0]]
            print("Tahmin edilen sınıf:", predicted_class_label)

            self.predicted_class_label = predicted_class_label

        except Exception as e:
            print(f'Hata oluştu: {str(e)}')


class FindMinutaeiesThread(QThread):
    finished_signal = pyqtSignal(list)

    def __init__(self, image_path):
        super().__init__()
        self.value_list = []
        self.image_path = image_path

    def run(self):
        self.findMinutaeies(self.image_path)
        self.finished_signal.emit(self.value_list)

    def findMinutaeies(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        kp1, des1 = Descriptors.getDescriptors(img)

        marked_img = cv2.drawKeypoints(img, kp1, outImage=None)

        self.value_list.append(img)
        self.value_list.append(marked_img)


class MatchFingerprintsThread(QThread):
    finished_signal = pyqtSignal(list)

    def __init__(self, img1, img2):
        super().__init__()
        self.value_list = []
        self.img1 = img1
        self.img2 = img2

    def run(self):
        self.matchFingerprints(self.img1, self.img2)
        self.finished_signal.emit(self.value_list)

    def matchFingerprints(self, img1, img2):
        kp1, des1 = Descriptors.getDescriptors(img1)
        kp2, des2 = Descriptors.getDescriptors(img2)

        # Matching between descriptors
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = sorted(bf.match(des1, des2), key=lambda match: match.distance)

        # Calculate score
        score = 0
        for match in matches:
            score += match.distance
        score_threshold = 33

        self.value_list.append(score / len(matches))

        if score / len(matches) < score_threshold:
            self.value_list.append("Parmak İzleri Eşleşti")
            print("Fingerprint matches.")
        else:
            self.value_list.append("Parmak İzleri Eşleşmedi")
            print("Fingerprint does not match.")

        # Plot matches
        # nitemleri eşleştir
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches, flags=2, outImg=None)
        self.value_list.append(img3)


class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setGeometry(400, 100, 1200, 800)
        self.setStyleSheet("background-color: rgb(169, 231, 235);")
        self.setWindowTitle("Parmak İzi Sınıflandırıcı")

        self.createButtons()
        self.createLabels()
        self.createProgressBars()

    def createButtons(self):
        self.selectImgButton = QPushButton("Parmak izi seç", self)
        self.selectImgButton.setGeometry(20, 370, 150, 70)
        self.selectImgButton.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.selectImgButton.clicked.connect(lambda: self.selectImg())

        self.predictButton = QPushButton("Sınıfını Bul", self)
        self.predictButton.setGeometry(200, 600, 150, 70)
        self.predictButton.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.predictButton.clicked.connect(lambda: self.predictThread(self.selected_image) if hasattr(self,
                                                                                                      'selected_image') else self.handleMissingImage())

        self.markMinutaeisButton = QPushButton("Nitemleri İşaretle", self)
        self.markMinutaeisButton.setGeometry(370, 600, 150, 70)
        self.markMinutaeisButton.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.markMinutaeisButton.clicked.connect(lambda: self.findMinutaeiesThread(self.selected_image) if hasattr(self,
                                                                                                                   'selected_image') else self.handleMissingImage())

        self.matchButton = QPushButton("Eşleştir", self)
        self.matchButton.setGeometry(540, 600, 150, 70)
        self.matchButton.setStyleSheet("background-color: rgb(255, 255, 255);")
        self.matchButton.clicked.connect(lambda: self.matchFingerprintsThread(self.selected_image) if hasattr(self,
                                                                                                              'selected_image') else self.handleMissingImage())

    def createLabels(self):
        font = QFont()
        font.setFamily("Arial")
        font.setPointSize(16)

        self.imageLabel = QLabel("Parmak İzi Fotoğrafı Seç", self)
        self.imageLabel.setGeometry(200, 50, 512, 512)
        self.imageLabel.setAlignment(Qt.AlignCenter)
        self.imageLabel.setStyleSheet("background-color: rgb(200, 200, 200); border-radius: 10px;")

        self.labelArch = QLabel("Arch", self)
        self.labelArch.setGeometry(1000, 100, 150, 100)
        self.labelArch.setFont(font)
        self.labelArch.setAlignment(Qt.AlignCenter)
        self.labelArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")

        self.labelLeftLoop = QLabel("Left Loop", self)
        self.labelLeftLoop.setGeometry(1000, 210, 150, 100)
        self.labelLeftLoop.setFont(font)
        self.labelLeftLoop.setAlignment(Qt.AlignCenter)
        self.labelLeftLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")

        self.labelRightLoop = QLabel("Right Loop", self)
        self.labelRightLoop.setGeometry(1000, 320, 150, 100)
        self.labelRightLoop.setFont(font)
        self.labelRightLoop.setAlignment(Qt.AlignCenter)
        self.labelRightLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")

        self.labelTentedArch = QLabel("Tented Arch", self)
        self.labelTentedArch.setGeometry(1000, 430, 150, 100)
        self.labelTentedArch.setFont(font)
        self.labelTentedArch.setAlignment(Qt.AlignCenter)
        self.labelTentedArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")

        self.labelWhorl = QLabel("Whorl", self)
        self.labelWhorl.setGeometry(1000, 540, 150, 100)
        self.labelWhorl.setFont(font)
        self.labelWhorl.setAlignment(Qt.AlignCenter)
        self.labelWhorl.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")

    def createProgressBars(self):
        progressbar = QProgressBar()
        progressbar.setMaximum(100)
        progressbarLayout = QVBoxLayout()
        progressbarLayout.addWidget(progressbar)
        progressbar.setValue(50)
        message = "Parmak izi İşleniyor..."
        self.msg_box = QMessageBox()
        self.msg_box.setIcon(QMessageBox.Warning)
        self.msg_box.setText("Parmak izi İşleniyor...")
        self.msg_box.setInformativeText(message)
        self.msg_box.setWindowTitle("")
        self.msg_box.setStandardButtons(QMessageBox.NoButton)

    def selectImg(self):
        file_dialog = QFileDialog(self)
        file_path, _ = file_dialog.getOpenFileName(self, 'Dosya Seç', '', 'All Files (*)')

        if file_path:
            pixmap = QPixmap(file_path)
            self.imageLabel.setPixmap(pixmap)

            self.selected_image = file_path
            return file_path

        else:
            return

    def predictThread(self, image_path):
        if not self.validateImagePath(image_path):
            return False

        try:
            self.predict_thread = PredictThread(image_path, image_width=224, image_height=224)
            self.predict_thread.finished_signal.connect(self.predictFinished)
            self.predict_thread.start()

            self.msg_box.show()

        except Exception as e:
            print(f"Error during prediction: {e}")

        return None

    def predictFinished(self, predicted_class_label):
        self.msg_box.reject()
        self.showPatern(predicted_class_label)

    def findMinutaeiesThread(self, image_path):
        if not self.validateImagePath(image_path):
            return False

        self.find_minutaeies_thread = FindMinutaeiesThread(image_path)
        self.find_minutaeies_thread.finished_signal.connect(self.findMinutaeiesFinished)
        self.find_minutaeies_thread.start()

        self.msg_box.show()

    def findMinutaeiesFinished(self, value_list):
        self.msg_box.reject()
        self.window_instance3 = ShowMinutaeies.App(value_list)
        self.window_instance3.show()

    def matchFingerprintsThread(self, selected_image):
        if not self.validateImagePath(selected_image):
            return False
        img1 = cv2.imread(selected_image, cv2.IMREAD_GRAYSCALE)

        image_path2 = self.selectImg()
        if not self.validateImagePath(image_path2):
            return False
        img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

        self.match_fingerprints_thread = MatchFingerprintsThread(img1, img2)
        self.match_fingerprints_thread.finished_signal.connect(self.matchFingerprintsFinished)
        self.match_fingerprints_thread.start()

        self.msg_box.show()

    def matchFingerprintsFinished(self, value_list):
        self.msg_box.reject()
        self.window_instance3 = ShowMatches.App(value_list)
        self.window_instance3.show()

    def validateImagePath(self, image_path):

        if not image_path:
            message = "Dosya seç"
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Hata")
            msg_box.setInformativeText(message)
            msg_box.setWindowTitle("Dosya Doğrulama")
            msg_box.exec_()
            return False

        if not os.path.isfile(image_path):
            message = "Belirtilen dosya bulunamadı."
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Hata")
            msg_box.setInformativeText(message)
            msg_box.setWindowTitle("Dosya Doğrulama")
            msg_box.exec_()
            return False

        _, file_extension = os.path.splitext(image_path)
        allowed_extensions = ['.jpg', '.jpeg', '.png']

        if file_extension.lower() not in allowed_extensions:
            message = "Dosya uzantısı desteklenmiyor. Sadece .jpg, .jpeg ve .png uzantıları kabul edilir."
            msg_box = QMessageBox()
            msg_box.setIcon(QMessageBox.Warning)
            msg_box.setText("Hata")
            msg_box.setInformativeText(message)
            msg_box.setWindowTitle("Dosya Doğrulama")
            msg_box.exec_()
            return False
        return True

    def handleMissingImage(self):
        QMessageBox.warning(self, "Uyarı", "Önce bir resim seçmelisiniz.")

    def showPatern(self, predicted_class_label):
        if predicted_class_label == "A":
            self.labelArch.setStyleSheet("background-color: rgb(52, 242, 10); border-radius: 10px;")
            self.labelLeftLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelRightLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelTentedArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelWhorl.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
        elif predicted_class_label == "L":
            self.labelArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelLeftLoop.setStyleSheet("background-color: rgb(52, 242, 10); border-radius: 10px;")
            self.labelRightLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelTentedArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelWhorl.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
        elif predicted_class_label == "R":
            self.labelArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelLeftLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelRightLoop.setStyleSheet("background-color: rgb(52, 242, 10); border-radius: 10px;")
            self.labelTentedArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelWhorl.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
        elif predicted_class_label == "T":
            self.labelArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelLeftLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelRightLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelTentedArch.setStyleSheet("background-color: rgb(52, 242, 10); border-radius: 10px;")
            self.labelWhorl.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
        elif predicted_class_label == "W":
            self.labelArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelLeftLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelRightLoop.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelTentedArch.setStyleSheet("background-color: rgb(255, 255, 255); border-radius: 10px;")
            self.labelWhorl.setStyleSheet("background-color: rgb(52, 242, 10); border-radius: 10px;")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())
