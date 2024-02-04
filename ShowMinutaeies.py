import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(121)
        self.axes2 = fig.add_subplot(122)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)
        fig.subplots_adjust(wspace=0.5)

    def plot(self, value_list):
        img = value_list[0]
        marked_image = value_list[1]

        # İlk alt grafikte orijinal görüntüyü göster
        self.axes1.imshow(img, cmap='gray')
        self.axes1.set_title('Asil Parmak izi')

        # İkinci alt grafikte işaretlenmiş görüntüyü göster
        self.axes2.imshow(marked_image, cmap='gray')
        self.axes2.set_title('İsaretlenmis Parmak izi')

        self.draw()


class App(QMainWindow):
    def __init__(self, value_list):
        super().__init__()
        self.title = 'Parmak izi ayrintilari'
        self.initUI(value_list)

    def initUI(self, value_list):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1000, 800)

        m = PlotCanvas(self, width=10, height=8)
        m.plot(value_list)

        self.show()
