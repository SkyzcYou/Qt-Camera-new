from PyQt5.QtCore import QThread,pyqtSignal

class WorkThread(QThread):
    # 使用信号和UI主线程通讯，参数是发送信号时附带参数的数据类型
    finishSignal = pyqtSignal(list)

    # 带参数实例
    def __init__(self,known_list_db,parent=None):
        super(WorkThread,self).__init__(parent)
        self.known_list_db = known_list_db
    def run(self):

        # 耗时操作
        pass

