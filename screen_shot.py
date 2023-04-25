import sys
from datetime import datetime
import imageio.v2 as imageio
from PyQt5.QtCore import Qt, QRect, QTimer, QEvent, QObject, QSize
from PyQt5.QtGui import QPixmap, QPainter, QColor
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QVBoxLayout
import numpy as np

"""
这个程序是一个截屏工具
"""

class ScreenshotWidget(QWidget, QObject):
    def __init__(self):
        """
        设置窗口的标题、大小、位置和属性。

        我们还创建了一个 QLabel 和三个 QPushButton，

        分别用于显示截屏、选择区域、开始/停止录制和关闭窗口。

        recording 变量用于记录是否正在录制，

        rect 变量用于记录选择的区域，

        images 变量用于存储录制的帧
        """
        super().__init__()
        self.setWindowTitle('Screenshot Widget')
        self.setGeometry(100, 100, 400, 300)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        # 创建控件
        self.screenshotLabel = QLabel(self)
        self.selectAreaButton = QPushButton('Select Area', self)
        self.recordButton = QPushButton('Start Recording', self)
        self.closeButton = QPushButton('Close', self)
        # 创建布局管理器
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        # 添加控件到布局管理器
        layout.addWidget(self.screenshotLabel, 1)
        layout.addStretch(1)
        layout.addWidget(self.selectAreaButton)
        layout.addWidget(self.recordButton)
        layout.addWidget(self.closeButton)
        # 设置控件属性
        self.screenshotLabel.setFixedSize(400, 250)
        self.selectAreaButton.setFixedSize(80, 30)
        self.recordButton.setFixedSize(80, 30)
        self.closeButton.setFixedSize(80, 30)
        # 连接信号和槽
        self.selectAreaButton.clicked.connect(self.selectArea)
        self.recordButton.clicked.connect(self.toggleRecording)
        self.closeButton.clicked.connect(self.close)
        # 初始化变量
        self.recording = False
        self.rect = None
        self.startPos = None
        self.endPos = None
        self.images = []
        self.last_press_time = datetime.now()
        self.mask = None
        self.maskPainter = None
        self.start = False
        self.end = False

    def selectArea(self):
        """
        方法用于选择截屏区域
        """
        print(f"selectArea start")
        self.recording = False
        self.rect = None
        self.startPos = None
        self.endPos = None
        self.images = []
        self.start = True
        self.end = False

    def toggleRecording(self):
        """
        开始/停止录制
        """
        self.recording = not self.recording

        if self.recording:
            print("开始录制")
            self.recordButton.setText('Stop Recording')
            self.images = []
            self.timer = QTimer()
            self.timer.timeout.connect(self.updateRecording)
            self.timer.start(100)
        else:
            print("停止录制")
            self.recordButton.setText('Start Recording')
            self.timer.stop()
            imageio.mimsave('screenshot.gif', self.images)

    def updateRecording(self):
        """
        更新录制的帧
        """
        now = datetime.now()
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f")
        print(f"更新录制->{timestamp}")
        screenshot = QApplication.primaryScreen().grabWindow(
            QApplication.desktop().winId(), self.rect.x(), self.rect.y(),
            self.rect.width(), self.rect.height())
        image = screenshot.toImage()
        print(f"image->{image}")
        h = image.height()
        w = image.width()
        bits = image.constBits()
        bits.setsize(image.byteCount())
        arr = np.array(bits).reshape(h, w, 4)
        self.images.append(arr)
        self.update()

    def keyPressEvent(self, event):
        """
        处理按键事件
        """
        if event.key() == Qt.Key_Return:
            self.toggleRecording()

    # def mousePressEvent(self, event):
    #     """
    #     鼠标按下
    #     """
    #     self.startPos = event.pos()
    #     print(f"startPos->{self.startPos}")

    # def mouseReleaseEvent(self, event):
    #     """
    #     鼠标释放
    #     """
    #     self.endPos = event.pos()
    #     print(f"endPos->{self.endPos}")
    #     self.rect = QRect(self.startPos, self.endPos).normalized()
    #     print(f"rect->{self.rect}")
    #     self.setMouseTracking(False) 

    def eventFilter(self, obj, event):
        if event.type() == QEvent.MouseButtonPress:
            """
            鼠标按下
            """
            if self.start:
                current_time = datetime.now()
            # if (current_time - self.last_press_time).total_seconds() < 1:
            #     return super().eventFilter(obj, event)
                self.last_press_time = current_time
                self.startPos = event.pos()
                print(f"startPos->{self.startPos}")
                # self.mask = QApplication.primaryScreen().grabWindow(QApplication.desktop().winId())
                # self.maskPainter = QPainter(self.mask)
                # self.maskPainter.setCompositionMode(QPainter.CompositionMode_Clear)
                # self.maskPainter.fillRect(self.mask.rect(), QColor(0, 0, 0, 0))
                # self.maskPainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                # self.maskPainter.fillRect(self.mask.rect(), QColor(0, 0, 0, 128))
                # self.maskPainter.setCompositionMode(QPainter.CompositionMode_Clear)
                # self.maskPainter.fillRect(QRect(self.startPos, QSize()), QColor(0, 0, 0, 0))
                # self.maskPainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
                # self.maskPainter.fillRect(QRect(self.startPos, QSize()), QColor(0, 0, 0, 128))
                # self.update()
        # elif event.type() == QEvent.MouseMove:
        #     if self.maskPainter is not None:
        #         if self.startPos is not None:
        #             if self.start:
        #                 self.endPos = event.pos()
        #                 self.maskPainter.setCompositionMode(QPainter.CompositionMode_Clear)
        #                 self.maskPainter.fillRect(QRect(self.startPos, self.endPos).normalized(), QColor(0, 0, 0, 0))
        #                 self.maskPainter.setCompositionMode(QPainter.CompositionMode_SourceOver)
        #                 self.maskPainter.fillRect(QRect(self.startPos, self.endPos).normalized(), QColor(0, 0, 0, 128))
        #                 self.update()
        #                 self.end = true
        elif event.type() == QEvent.MouseButtonRelease:
            """
            鼠标释放
            """
            if self.start:
            # current_time = datetime.now()
            # if (current_time - self.last_press_time).total_seconds() < 1:
            #     return super().eventFilter(obj, event)
            # self.last_press_time = current_time
                self.endPos = event.pos()
                print(f"endPos->{self.endPos}")
                self.rect = QRect(self.startPos, self.endPos).normalized()
                print(f"rect->{self.rect}")
                self.start = False
                # self.maskPainter.end()
                self.update()
        return super().eventFilter(obj, event)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    widget = ScreenshotWidget()
    widget.show()
    app.installEventFilter(widget)
    sys.exit(app.exec_())