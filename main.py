import os
import sys
import torchvision.utils
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QMutex, QCoreApplication
from PyQt5.QtGui import QPixmap, QMovie, QIcon, QImage
from PyQt5.QtWidgets import *
from torch.autograd import Variable
from torchvision.utils import save_image
from ntpath import basename
from net.fusion import GeneratorNet
from torchvision import transforms
from PIL import Image
import shutil
import torch
import cv2
import numpy as np

def getImage(filePath):
    img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (256, 256))
    return img

class MainWindow(QWidget):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.setFixedSize(720, 344)
        self.move(300, 300)
        self.setWindowTitle("Enhance")

        self.label1 = QLabel(self)
        self.label1.setFixedSize(256, 256)
        self.label1.move(160, 22)
        self.label1.setStyleSheet("QLabel{background:white;}")

        self.label2 = QLabel(self)
        self.label2.setFixedSize(256, 256)
        self.label2.move(436, 22)
        self.label2.setStyleSheet("QLabel{background:white;}")

        self.label3 = QLabel(self)
        self.label3.setAlignment(Qt.AlignCenter)
        self.label3.setText("Original")
        self.label3.setFixedSize(256, 22)
        self.label3.move(160, 277)

        self.label4 = QLabel(self)
        self.label4.setAlignment(Qt.AlignCenter)
        self.label4.setText("Enhanced")
        self.label4.setFixedSize(256, 22)
        self.label4.move(436, 277)

        self.btn1 = QPushButton("Open Image", self)
        self.btn1.resize(140, 30)
        self.btn1.move(12, 22)
        self.btn1.clicked.connect(self.open)

        self.btn2 = QPushButton("Enhance", self)
        self.btn2.resize(140, 30)
        self.btn2.move(12, 44)
        self.btn2.clicked.connect(self.enhance)
        self.btn2.setEnabled(False)

        self.btn3 = QPushButton("Enhance(In Batch)", self)
        self.btn3.resize(140, 30)
        self.btn3.move(12, 66)
        self.btn3.setIcon(QIcon())
        self.animation = QMovie("./icon/processing.gif")
        self.animation.frameChanged.connect(self.updateAniamation)
        self.btn3.clicked.connect(self.batch_enhance)

        self.btn4 = QPushButton("Save Result", self)
        self.btn4.resize(140, 30)
        self.btn4.move(12, 88)
        self.btn4.clicked.connect(self.save)
        self.btn4.setEnabled(False)

        self.btn5 = QPushButton("< Previous", self)
        self.btn5.resize(140, 30)
        self.btn5.move(284, 305)
        self.btn5.clicked.connect(self.previous)
        self.btn5.setEnabled(False)

        self.btn6 = QPushButton("Next >", self)
        self.btn6.resize(140, 30)
        self.btn6.move(428, 305)
        self.btn6.clicked.connect(self.next)
        self.btn6.setEnabled(False)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignRight)
        if torch.cuda.is_available():
            self.label.setText("Device : CUDA")
        else:
            self.label.setText("Device : CPU")
        self.label.setFixedSize(256, 22)
        self.label.move(436, 5)

        self.imgName = ""
        self.dataset = ""
        self.index = 0
        self.files = []
        self.openThread = OpenThread()
        self.openThread.signal.connect(self.open_callback)
        self.openThread.start()
        self.enhanceThread = EnhanceThread()
        self.enhanceThread.signal1.connect(self.enhance_callback)
        self.enhanceThread.signal2.connect(self.batch_callback)
        self.enhanceThread.start()

        self.img = None
        self.enhanced = None

    def updateAniamation(self):
        self.btn3.setIcon(QIcon(self.animation.currentPixmap()))

    def open(self):
        imgName, _ = QFileDialog.getOpenFileName(self, "Open...", "", "*.jpg | *.png")
        if imgName != "":
            self.imgName = imgName
            self.label1.clear()
            self.label2.clear()
            self.img = None
            self.enhanced = None
            self.dataset = ""
            self.index = 0
            self.files = []
            self.btn4.setEnabled(False)
            self.btn5.setEnabled(False)
            self.btn6.setEnabled(False)
            self.img = getImage(self.imgName)
            self.openThread.img = self.img
            self.openThread.unlock()

    def enhance(self):
        self.enhanceThread.img = Image.fromarray(self.img)
        self.enhanceThread.unlock()

    def batch_enhance(self):
        path = QFileDialog.getExistingDirectory(self, "Open Directory")
        if path != "":
            if not os.path.exists("./enhance"):
                os.mkdir("./enhance")
            self.label1.clear()
            self.label2.clear()
            self.dataset = ""
            self.index = 0
            self.files = []
            self.dataset = path
            self.btn4.setEnabled(False)
            for i in os.listdir(path):
                if i.split(".")[-1] in ["jpg", "png"]:
                    self.files.append(i)
            self.btn1.setEnabled(False)
            self.btn2.setEnabled(False)
            self.btn3.setEnabled(False)
            self.btn3.setText("")
            self.animation.start()
            self.enhanceThread.path = path
            self.enhanceThread.files = self.files
            self.enhanceThread.unlock()

    def save(self):
        if self.enhanced is not None:
            path = QFileDialog.getSaveFileName(self, "Save", self.imgName[0:-4] + "_enhanced", "*.png")
            if path[0] != "":
                self.enhanced.copy().save(path[0])

    def open_callback(self, img):
        self.label1.setPixmap(img)
        self.btn2.setEnabled(True)

    def enhance_callback(self, qpixmap):
        self.enhanced = qpixmap
        self.label2.setPixmap(qpixmap)
        self.btn2.setEnabled(False)
        self.btn4.setEnabled(True)

    def batch_callback(self):
        if len(self.files) > 1:
            self.btn6.setEnabled(True)
        self.btn3.setEnabled(True)
        self.btn1.setEnabled(True)
        self.animation.stop()
        self.btn3.setText("Enhance(In Batch)")
        self.btn3.setIcon(QIcon())

        origin = getImage(os.path.join(self.dataset, self.files[0]))
        origin = QPixmap.fromImage(QImage(origin, 256, 256, QImage.Format_RGB888))

        enhance = getImage(os.path.join("./enhance", self.files[0]))
        enhance = QPixmap.fromImage(QImage(enhance, 256, 256, QImage.Format_RGB888))

        self.label1.setPixmap(origin)
        self.label2.setPixmap(enhance)

    def previous(self):
        self.index = self.index - 1
        if self.index == 0:
            self.btn5.setEnabled(False)
        elif self.index == len(self.files) - 2:
            self.btn6.setEnabled(True)

        origin = getImage(os.path.join(self.dataset, self.files[self.index]))
        origin = QPixmap.fromImage(QImage(origin, 256, 256, QImage.Format_RGB888))

        enhance = getImage(os.path.join("./enhance", self.files[self.index]))
        enhance = QPixmap.fromImage(QImage(enhance, 256, 256, QImage.Format_RGB888))

        self.label1.setPixmap(origin)
        self.label2.setPixmap(enhance)

    def next(self):
        self.index = self.index + 1
        if self.index == 1:
            self.btn5.setEnabled(True)
        elif self.index - len(self.files) == -1:
            self.btn6.setEnabled(False)

        origin = getImage(os.path.join(self.dataset, self.files[self.index]))
        origin = QPixmap.fromImage(QImage(origin, 256, 256, QImage.Format_RGB888))

        enhance = getImage(os.path.join("./enhance", self.files[self.index]))
        enhance = QPixmap.fromImage(QImage(enhance, 256, 256, QImage.Format_RGB888))

        self.label1.setPixmap(origin)
        self.label2.setPixmap(enhance)

    def closeEvent(self, event):
        if os.path.exists("./enhance"):
            reply = QMessageBox.question(self, "Warning", "Do you want to keep the enhanced results？", QMessageBox.Yes, QMessageBox.No)
            if reply == QMessageBox.Yes:
                pass
            else:
                shutil.rmtree("./enhance")


class OpenThread(QThread):
    signal = pyqtSignal(QPixmap)

    def __init__(self):
        super(OpenThread, self).__init__()
        self.img = None
        self.qmutex = QMutex()

    def __del__(self):
        self.wait()

    def unlock(self):
        self.qmutex.unlock()

    def run(self):
        while True:
            self.qmutex.lock()
            if self.img is not None:
                img = QPixmap.fromImage(QImage(self.img, 256, 256, QImage.Format_RGB888))
                self.signal.emit(img)


class EnhanceThread(QThread):
    signal1 = pyqtSignal(QPixmap)
    signal2 = pyqtSignal()

    def __init__(self):
        super(EnhanceThread, self).__init__()
        if torch.cuda.is_available():
            self.model = GeneratorNet().cuda()
            self.Tensor = torch.cuda.FloatTensor
            self.model.load_state_dict(torch.load("./model/generator.pth"))
        else:
            self.model = GeneratorNet()
            self.Tensor = torch.FloatTensor
            self.model.load_state_dict(torch.load("./model/generator.pth", map_location=torch.device('cpu')))
        self.model = self.model.eval()

        self.transforms_ = [transforms.Resize((256, 256), Image.BICUBIC),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), ]
        self.transform = transforms.Compose(self.transforms_)
        self.img = None
        self.path = ""
        self.files = []
        self.qmutex = QMutex()

    def __del__(self):
        self.wait()

    def unlock(self):
        self.qmutex.unlock()

    def run(self):
        while True:
            self.qmutex.lock()
            if self.path == "":
                if self.img is not None:
                    inp = self.transform(self.img)
                    inp = Variable(inp).type(self.Tensor).unsqueeze(0)
                    output = self.model(inp)
                    grid = torchvision.utils.make_grid(output, normalize=True)
                    output = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    output = QImage(output.tobytes(), 256, 256, QImage.Format_RGB888)
                    output = QPixmap.fromImage(output)
                    self.signal1.emit(output)
                    self.img = None
            else:
                for file in self.files:
                    inp = Image.open(os.path.join(self.path, file))
                    inp = self.transform(inp)
                    inp = Variable(inp).type(self.Tensor).unsqueeze(0)
                    output = self.model(inp)
                    save_image(output.data, os.path.join("./enhance", basename(file)), normalize=True)
                self.signal2.emit()
                self.path = ""
                self.files = []


if __name__ == '__main__':
    QCoreApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
