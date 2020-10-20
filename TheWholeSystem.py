from PyQt5 import QtCore, QtGui, QtWidgets
from sys import argv,exit
from PyQt5.QtWidgets import QApplication, QMainWindow, QDialog, QPushButton, QInputDialog
import time
import cv2
import test
from util.load_knowface import load_face_data
from util import detect_face,collection_face_image
from smbus2 import SMBus
from mlx90614 import MLX90614
import face_recognition

class Ui_MainWindow(object):


    def __init__(self, MainWindow):
        '''
        启动程序时，加载本地已知人脸数据库
        load_face_data
        :return:
        known_list_db[] 列表有两个元素，0是已知人脸的编码数组，1是姓名
        '''
        # 定义全局变量
        self.KNOWN_LIST_DB = load_face_data()

        self.timer_camera = QtCore.QTimer() # 定时器
        self.setupUi(MainWindow)
        self.retranslateUi(MainWindow)
        self.cap = cv2.VideoCapture() # 准备获取图像
        self.CAM_NUM = 0
        self.slot_init() # 设置槽函数

    def setupUi(self, MainWindow):
        """windows interface"""
        MainWindow.setObjectName("MainWindow")
        MainWindow.setWindowFlags(QtCore.Qt.CustomizeWindowHint)  # 去掉标题栏的代码
        MainWindow.setWindowModality(QtCore.Qt.NonModal)
        MainWindow.resize(765, 645)
        MainWindow.setMinimumSize(QtCore.QSize(765, 645))
        MainWindow.setMaximumSize(QtCore.QSize(16777215, 16777215))
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(":/newPrefix/pic/pai.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        MainWindow.setToolTip("")
        MainWindow.setAutoFillBackground(False)
        MainWindow.setTabShape(QtWidgets.QTabWidget.Rounded)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.verticalLayout = QtWidgets.QVBoxLayout()
        self.verticalLayout.setObjectName("verticalLayout")
        """app name"""
        self.label_title = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.MinimumExpanding, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_title.sizePolicy().hasHeightForWidth())
        self.label_title.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setFamily("华文隶书")
        font.setPointSize(20)
        self.label_title.setFont(font)
        self.label_title.setAlignment(QtCore.Qt.AlignCenter)
        self.label_title.setObjectName("label_title")
        self.verticalLayout.addWidget(self.label_title)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.horizontalLayout.setContentsMargins(-1, 20, -1, -1)
        self.horizontalLayout.setSpacing(0)
        self.horizontalLayout.setObjectName("horizontalLayout")
        """open camera"""
        self.pushButton_open = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_open.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_open.setMaximumSize(QtCore.QSize(120, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_open.setFont(font)
        icon1 = QtGui.QIcon()
        icon1.addPixmap(QtGui.QPixmap(":/newPrefix/pic/g1.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_open.setIcon(icon1)
        self.pushButton_open.setObjectName("pushButton_open")
        self.horizontalLayout.addWidget(self.pushButton_open)
        """input info"""
        self.pushButton_info = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.pushButton_info.sizePolicy().hasHeightForWidth())
        self.pushButton_info.setSizePolicy(sizePolicy)
        self.pushButton_info.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_info.setMaximumSize(QtCore.QSize(100, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_info.setFont(font)
        self.pushButton_info.setIcon(icon)
        self.pushButton_info.setObjectName("pushButton_info")
        self.horizontalLayout.addWidget(self.pushButton_info)
        """close camera"""
        self.pushButton_close = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton_close.setMinimumSize(QtCore.QSize(100, 40))
        self.pushButton_close.setMaximumSize(QtCore.QSize(130, 40))
        font = QtGui.QFont()
        font.setFamily("华文彩云")
        font.setPointSize(12)
        self.pushButton_close.setFont(font)
        icon2 = QtGui.QIcon()
        icon2.addPixmap(QtGui.QPixmap(":/newPrefix/pic/down.png"), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.pushButton_close.setIcon(icon2)
        self.pushButton_close.setObjectName("pushButton_close")
        self.horizontalLayout.addWidget(self.pushButton_close)
        """box of name mask temperature"""
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setContentsMargins(160, -1, 0, -1)
        self.verticalLayout_2.setSpacing(6)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        """name"""
        self.label_name = QtWidgets.QLabel(self.centralwidget)
        self.label_name.setObjectName("label_name")
        self.verticalLayout_2.addWidget(self.label_name)
        """temperature"""
        self.label_temperature = QtWidgets.QLabel(self.centralwidget)
        self.label_temperature.setObjectName("label_temperature")
        self.verticalLayout_2.addWidget(self.label_temperature)
        """ismask"""
        self.label_ismask = QtWidgets.QLabel(self.centralwidget)
        self.label_ismask.setObjectName("label_ismask")
        self.verticalLayout_2.addWidget(self.label_ismask)
        self.horizontalLayout.addLayout(self.verticalLayout_2)
        self.horizontalLayout.setStretch(0, 1)
        self.horizontalLayout.setStretch(1, 1)
        self.verticalLayout.addLayout(self.horizontalLayout)
        """video flow"""
        self.label_face = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_face.sizePolicy().hasHeightForWidth())
        self.label_face.setSizePolicy(sizePolicy)
        self.label_face.setMinimumSize(QtCore.QSize(0, 0))
        self.label_face.setMaximumSize(QtCore.QSize(16777215, 16777215))
        font = QtGui.QFont()
        font.setFamily("楷体")
        font.setPointSize(16)
        self.label_face.setFont(font)
        self.label_face.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.label_face.setStyleSheet("background-color: rgb(192, 218, 255);")
        self.label_face.setAlignment(QtCore.Qt.AlignCenter)
        self.label_face.setObjectName("label_face")
        self.verticalLayout.addWidget(self.label_face)
        self.verticalLayout.setStretch(2, 5)
        self.horizontalLayout_2.addLayout(self.verticalLayout)
        MainWindow.setCentralWidget(self.centralwidget)
        """unknown"""
        self.actionGoogle_Translate = QtWidgets.QAction(MainWindow)
        self.actionGoogle_Translate.setObjectName("actionGoogle_Translate")
        self.actionHTML_type = QtWidgets.QAction(MainWindow)
        self.actionHTML_type.setObjectName("actionHTML_type")
        self.actionsoftware_version = QtWidgets.QAction(MainWindow)
        self.actionsoftware_version.setObjectName("actionsoftware_version")

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    """"设置槽函数 （设置点击事件）"""
    def slot_init(self):
        """点击打开相机"""
        self.pushButton_open.clicked.connect(self.button_open_camera_click)
        """显示视频画面"""
        self.timer_camera.timeout.connect(self.show_camera)
        """关闭摄像头按钮点击事件"""
        self.pushButton_close.clicked.connect(self.button_close_camera_click)
        """录入信息按钮点击事件"""
        self.pushButton_info.clicked.connect(self.dialog_window)
    """
    实现打开相机方法
    """
    def button_open_camera_click(self):
        if self.timer_camera.isActive() == False:
            flag = self.cap.open(self.CAM_NUM)
            if flag == False:
                msg = QtWidgets.QMessageBox.warning(
                    self, u"Warning", u"请检测相机是否连接正确",
                    buttons=QtWidgets.QMessageBox.Ok,
                    defaultButton=QtWidgets.QMessageBox.Ok)
            else:
                self.timer_camera.start(40) # 定时器开始计时30ms，结果是每过30ms从摄像头中取一帧显示

    """
    实现显示视频流
        -> 开始识别人脸
    """
    def show_camera(self):
        flag, self.image = self.cap.read()

        # 拿到视频帧
        no_ready_img = self.image
        self.image=cv2.flip(self.image, 1) # 左右翻转
        show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)

        self.label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
        self.label_face.setScaledContents(True)

        ''' 调用 detect_face() 方法识别人脸 '''
        name = detect_face.get_detect_face(no_ready_img,known_list_db=self.KNOWN_LIST_DB)
        print(name)
        # 设置 Lable 为姓名
        if name == None:
            self.label_name.setText("未检测人脸数据")
        elif name == "Unknown":
            self.label_name.setText("警告！！！未认证身份")
            self.label_name.setStyleSheet("border-width:1px;border-style:solid;border-color:rgb(255,0,0);background-color:yellow")

        else:
            self.label_name.setText(name)
        # 设置温度
        bus = SMBus(1)
        sensor = MLX90614(bus, address=0x5A)
        temperature = sensor.get_object_1() + 3
        if temperature > 46:
            self.label_temperature.setText("警报!! 温度超标： "+str(round(temperature,2)))
            self.label_temperature.setStyleSheet("border-width:1px;border-style:solid;border-color:rgb(255,0,0);background-color:yellow")
        elif temperature < 30:
            self.label_temperature.setText("警报!! 温度过低： " + str(round(temperature, 2)))
            self.label_temperature.setStyleSheet(
                "border-width:1px;border-style:solid;border-color:rgb(255,0,0);background-color:yellow")
        else:
            self.label_temperature.setText(str(round(temperature,2)) + "温度正常范围（30～46）")
            self.label_temperature.setStyleSheet("border-width:1px;border-style:solid;border-color:rgb(32,14,69);background-color:green")

        bus.close()
    """实现录入信息功能"""
    """采集框"""
    def dialog_window(self):
        if self.timer_camera.isActive() != False:
            cv2.putText(self.image, 'Face_Info_Collection...',
                        (int(self.image.shape[1] / 2 - 130), int(self.image.shape[0] / 2)),
                        cv2.FONT_HERSHEY_SCRIPT_COMPLEX,
                        1.0, (255, 0, 0), 1)

            text, ok = QInputDialog.getText(None, '信息采集', '请输入你的姓名（拼音），采集即将完成请保持面部在镜头前^_^')
            if ok and text:
                # 开始采集
                print('开始采集啦。。。。Name:' + text)
                self.timer_camera.stop()
                # 保存文件
                collection_face_image.save_image(text, self.image)
                # 添加编码到数组

                # 将视频帧调整为1/4大小，以便更快地进行人脸识别处理
                small_frame = cv2.resize(self.image, (0, 0), fx=0.25, fy=0.25)
                # 将图像从BGR颜色（OpenCV使用）转换为RGB颜色（人脸识别使用）
                rgb_small_frame = small_frame[:, :, ::-1]

                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_code = face_recognition.face_encodings(rgb_small_frame, face_locations)

                print('befor:')
                print(len(self.KNOWN_LIST_DB[0]))

                self.KNOWN_LIST_DB[0].append(face_code)
                self.KNOWN_LIST_DB[1].append(text)

                print('after')
                print(len(self.KNOWN_LIST_DB[0]))



            show = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)  # 左右翻转
            showImage = QtGui.QImage(show.data, show.shape[1], show.shape[0], QtGui.QImage.Format_RGB888)
            self.label_face.setPixmap(QtGui.QPixmap.fromImage(showImage))
            self.label_face.setScaledContents(True)
        else:
            print('相机未打开')

    def keyPressEven(self,event):
        if event.key() == QtCore.Qt.Key_Q:
            self.slotLo



    """关闭摄像头功能"""
    def button_close_camera_click(self):
        if self.timer_camera.isActive() != False:
            ok = QtWidgets.QPushButton()
            cacel = QtWidgets.QPushButton()
            msg = QtWidgets.QMessageBox(QtWidgets.QMessageBox.Warning, u"关闭", u"是否关闭！")
            msg.addButton(ok,QtWidgets.QMessageBox.ActionRole)
            msg.addButton(cacel, QtWidgets.QMessageBox.RejectRole)
            ok.setText(u'确定')
            cacel.setText(u'取消')
            if msg.exec_() != QtWidgets.QMessageBox.RejectRole:
                if self.cap.isOpened():
                    self.cap.release()
                if self.timer_camera.isActive():
                    self.timer_camera.stop()
                self.label_face.setText("<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>")

    """设置部件文字"""
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "AWESOME-COOL"))
        self.label_title.setText(_translate("MainWindow", "AWESOME-COOL"))
        self.pushButton_open.setToolTip(_translate("MainWindow", "点击打开摄像头"))
        self.pushButton_open.setText(_translate("MainWindow", "打开摄像头"))
        self.pushButton_info.setToolTip(_translate("MainWindow", "点击拍照"))
        self.pushButton_info.setText(_translate("MainWindow", "录入信息"))
        self.pushButton_close.setToolTip(_translate("MainWindow", "点击关闭摄像头"))
        self.pushButton_close.setText(_translate("MainWindow", "关闭摄像头"))
        self.label_name.setText(_translate("MainWindow", "未识别(请打开摄像头)"))
        self.label_temperature.setText(_translate("MainWindow", "请打开摄像头启动系统"))
        self.label_ismask.setText(_translate("MainWindow", "awesome-cool 安全检测系统^_^"))
        self.label_face.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><img src=\":/newPrefix/pic/Hint.png\"/><span style=\" font-size:28pt;\">点击打开摄像头</span><br/></p></body></html>"))
        self.actionGoogle_Translate.setText(_translate("MainWindow", "Google Translate"))
        self.actionHTML_type.setText(_translate("MainWindow", "HTML type"))
        self.actionsoftware_version.setText(_translate("MainWindow", "software version"))




if __name__ == '__main__':
    app = QApplication(argv)
    window = QMainWindow()
    ui = Ui_MainWindow(window)
    window.showFullScreen() # full screen
    window.show()
    exit(app.exec_())
