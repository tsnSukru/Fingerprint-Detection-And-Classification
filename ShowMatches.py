import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class PlotCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes1 = fig.add_subplot(111)

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QSizePolicy.Expanding,
                                   QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def plot(self, value_list):
        img = value_list[2]

        # Show first fingerprint
        self.axes1.imshow(img, cmap='gray')
        title = "Eşleşme Sonucu: " + str(value_list[1]) + ", Sapma Oranı: " + str(value_list[0])
        self.axes1.set_title(title)

        self.draw()


class App(QMainWindow):
    def __init__(self, value_list):
        super().__init__()
        self.title = 'Parmak Eşleşmesi'
        self.initUI(value_list)

    def initUI(self, value_list):
        self.setWindowTitle(self.title)
        self.setGeometry(100, 100, 1000, 800)

        m = PlotCanvas(self, width=10, height=8)
        m.plot(value_list)

        self.show()
