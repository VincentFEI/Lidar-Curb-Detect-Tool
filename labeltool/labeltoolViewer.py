
import os
import sys
import mat4py
import numpy as np
import scipy.io as scio

from PyQt4 import QtGui, QtCore
from matplotlib.backends.backend_qt4agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt4agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D

import autolabel as AL



# 所有的代码都实现在这个类的内部
class LabelTool(QtGui.QMainWindow):

    # 初始化,初始化UI界面,初始化类变量
    def __init__(self):
        super(LabelTool, self).__init__()

        self.initUI()

        ## 内部数据变量
        self.read_path = None
        self.files_list = None
        self.files_num = None
        self.current_file = None # 当前处理的文件名称
        self.frame_id = None     # 当前处理的文件帧数
        self.layer_id = None     # 当前处理的雷达线数
        self.point_id = None     # 当前处理的雷达点数
        self.category = 0
        # 雷达数据
        self.lidar_data = None
        # 当前处理的那一线的雷达数据
        self.current_layer_data = None
        self.max_layer_num = 100
        self.max_point_num = 100
        self.transform_matrix_local = None
        self.transform_matrix_global = None

        ## 窗口大小(用于显示)
        self.win_size = 15

        ## 显示变量
        self.point3D_show = None
        self.pointHei_show = None
        self.pointInt_show = None

        ## 存储标注数据的表格
        self.result_table = None

    #####
    ## 初始化UI
    #####

    def initUI(self):
        self.statusBar()
        self.init_Toolbar()
        self.init_Layout()
        self.init_Shortcut()
        self.setGeometry(30, 30, 1920, 1080)
        self.setWindowTitle('Label Tool')
        self.show()

    # 初始化工具栏
    def init_Toolbar(self):
        self.toolbar = self.addToolBar('Tools')
        ## load file
        loadAction = QtGui.QAction(QtGui.QIcon('icon/open.png'), '&Tools', self)
        loadAction.setToolTip('open')
        loadAction.setShortcut('Ctrl+1')
        loadAction.triggered.connect(self._loadLidarFrame)
        self.toolbar.addAction(loadAction)
        ## select frame
        selectFrameAction = QtGui.QAction(QtGui.QIcon('icon/frame.png'), '&Tools', self)
        selectFrameAction.setToolTip('select frame')
        selectFrameAction.setShortcut('Ctrl+2')
        selectFrameAction.triggered.connect(self._selectFrame)
        self.toolbar.addAction(selectFrameAction)
        ## back frame
        backFrameAction = QtGui.QAction(QtGui.QIcon('icon/back.png'), '&Tools', self)
        backFrameAction.setToolTip('back frame')
        backFrameAction.setShortcut('Ctrl+3')
        backFrameAction.triggered.connect(self._backFrame)
        self.toolbar.addAction(backFrameAction)
        ## next frame
        nextFrameAction = QtGui.QAction(QtGui.QIcon('icon/next.png'), '&Tools', self)
        nextFrameAction.setToolTip('next frame')
        nextFrameAction.setShortcut('Ctrl+4')
        nextFrameAction.triggered.connect(self._nextFrame)
        self.toolbar.addAction(nextFrameAction)
        ## select layer
        selectLayerAction = QtGui.QAction(QtGui.QIcon('icon/layer.png'), '&Tools', self)
        selectLayerAction.setToolTip('select layer')
        selectLayerAction.setShortcut('Ctrl+5')
        selectLayerAction.triggered.connect(self._selectLayer)
        self.toolbar.addAction(selectLayerAction)
        ## back layer
        backLayerAction = QtGui.QAction(QtGui.QIcon('icon/back.png'), '&Tools', self)
        backLayerAction.setToolTip('back layer')
        backLayerAction.setShortcut('Ctrl+6')
        backLayerAction.triggered.connect(self._backLayer)
        self.toolbar.addAction(backLayerAction)
        ## next layer
        nextLayerAction = QtGui.QAction(QtGui.QIcon('icon/next.png'), '&Tools', self)
        nextLayerAction.setToolTip('next layer')
        nextLayerAction.setShortcut('Ctrl+7')
        nextLayerAction.triggered.connect(self._nextLayer)
        self.toolbar.addAction(nextLayerAction)
        ## auto label
        autoLabelAction = QtGui.QAction(QtGui.QIcon('icon/auto.png'), '&Tools', self)
        autoLabelAction.setToolTip('auto label')
        autoLabelAction.setShortcut('Ctrl+A')
        autoLabelAction.triggered.connect(self._autoLabel)
        self.toolbar.addAction(autoLabelAction)
        ## save
        saveAction = QtGui.QAction(QtGui.QIcon('icon/save.png'), '&Tools', self)
        saveAction.setToolTip('save label result')
        saveAction.setShortcut('Ctrl+S')
        saveAction.triggered.connect(self._save)
        self.toolbar.addAction(saveAction)
        ## exit
        exitAction = QtGui.QAction(QtGui.QIcon('icon/exit.png'), '&Tools', self)
        exitAction.setToolTip('exit')
        exitAction.setShortcut('Ctrl+E')
        exitAction.triggered.connect(self.close)
        self.toolbar.addAction(exitAction)

    # 初始化界面布局
    def init_Layout(self):
        ## 帧数,线数,点数显示控件
        frameLabelWidget = QtGui.QLabel('Frame : ')
        frameLabelWidget.setFont(QtGui.QFont('Monospace', 16 ))
        self.frameIdWidget = QtGui.QLabel('None')
        self.frameIdWidget.setFont(QtGui.QFont('Monospace', 16 ))
        layerLabelWidget = QtGui.QLabel('Layer : ')
        layerLabelWidget.setFont(QtGui.QFont('Monospace', 16))
        self.layerIdWidget = QtGui.QLabel('None')
        self.layerIdWidget.setFont(QtGui.QFont('Monospace', 16))
        pointLabelWidget = QtGui.QLabel('Point : ')
        pointLabelWidget.setFont(QtGui.QFont('Monospace', 16))
        self.pointIdWidget = QtGui.QLabel('None')
        self.pointIdWidget.setFont(QtGui.QFont('Monospace', 16))

        dispHbox = QtGui.QHBoxLayout()
        dispHbox.addWidget(frameLabelWidget)
        dispHbox.addWidget(self.frameIdWidget)
        dispHbox.addWidget(layerLabelWidget)
        dispHbox.addWidget(self.layerIdWidget)
        dispHbox.addWidget(pointLabelWidget)
        dispHbox.addWidget(self.pointIdWidget)

        ## 滑动条,按钮控件
        self.pointSliderWidget = QtGui.QSlider(QtCore.Qt.Horizontal, self)
        self.pointSliderWidget.setTickPosition(QtGui.QSlider.TicksBelow)
        self.pointSliderWidget.setMaximum(10)
        self.pointSliderWidget.setMinimum(0)

        nextButtonWidget = QtGui.QPushButton('Next Point')
        nextButtonWidget.setFont(QtGui.QFont('Monospace', 16))
        prevButtonWidget = QtGui.QPushButton('Prev Point')
        prevButtonWidget.setFont(QtGui.QFont('Monospace', 16))

        pointVbox = QtGui.QVBoxLayout()
        pointVbox.addWidget(self.pointSliderWidget)
        pointHbox = QtGui.QHBoxLayout()
        pointHbox.addWidget(prevButtonWidget)
        pointHbox.addWidget(nextButtonWidget)
        pointVbox.addLayout(pointHbox)

        ## 下拉菜单,按钮,表格控件
        classLabelWidget = QtGui.QLabel('Select Category')
        classLabelWidget.setFont(QtGui.QFont('Monospace', 16))
        self.classComboWidget = QtGui.QComboBox(self)
        self.classComboWidget.addItem('0.road')
        self.classComboWidget.addItem('1.curb')
        self.classComboWidget.addItem('2.noise')
        self.classComboWidget.setFont(QtGui.QFont('Monospace', 16))

        addButtonWidget = QtGui.QPushButton('&Add')
        addButtonWidget.setFont(QtGui.QFont('Monospace', 16))
        delButtonWidget = QtGui.QPushButton('&Del')
        delButtonWidget.setFont(QtGui.QFont('Monospace', 16))

        self.resultTableWidget = QtGui.QTableWidget(self)
        self.resultTableWidget.setFont(QtGui.QFont('Monospace', 16))
        self.resultTableWidget.setRowCount(50)
        self.resultTableWidget.setColumnCount(3)
        self.resultTableWidget.setColumnWidth(0, 160)
        self.resultTableWidget.setColumnWidth(1, 160)
        self.resultTableWidget.setColumnWidth(2, 160)
        self.resultTableWidget.setHorizontalHeaderLabels(['Start point', 'End point', 'Category'])
        self.resultTableWidget.setMinimumWidth(535)

        labelVbox = QtGui.QVBoxLayout()
        labelVbox.addWidget(classLabelWidget, alignment=QtCore.Qt.AlignHCenter)
        labelVbox.addWidget(self.classComboWidget)
        labelHbox = QtGui.QHBoxLayout()
        labelHbox.addWidget(addButtonWidget)
        labelHbox.addWidget(delButtonWidget)
        labelVbox.addLayout(labelHbox)
        labelVbox.addWidget(self.resultTableWidget)

        PanelVBox = QtGui.QVBoxLayout()
        PanelVBox.addLayout(pointVbox)
        PanelVBox.addLayout(labelVbox)
        PanelHbox = QtGui.QHBoxLayout()
        PanelHbox.addLayout(PanelVBox)
        PanelHbox.addStretch(1)

        ## 图像显示控件
        self.fig3D = Figure((5,3), 100)
        self.canvas3D = FigureCanvas(self.fig3D)
        self.toolbar3D = NavigationToolbar(self.canvas3D, self)
        self.axis3D = Axes3D(self.fig3D)


        self.figHei = Figure((5, 3), 100)
        self.canvasHei = FigureCanvas(self.figHei)
        self.axisHei = self.figHei.add_subplot(111)

        self.figWin = Figure((5,3), 100)
        self.canvasWin = FigureCanvas(self.figWin)
        self.axisWin = Axes3D(self.figWin)

        self.figInt = Figure((5,3), 100)
        self.canvasInt = FigureCanvas(self.figInt)
        self.axisInt = self.figInt.add_subplot(111)

        canvasVbox = QtGui.QVBoxLayout()
        canvasVbox.addWidget(self.canvas3D, stretch=5)
        canvasVbox.addWidget(self.toolbar3D, alignment=QtCore.Qt.AlignHCenter)
        canvasHbox = QtGui.QHBoxLayout()
        canvasHbox.addWidget(self.canvasHei)
        canvasHbox.addWidget(self.canvasWin)
        canvasHbox.addWidget(self.canvasInt)
        canvasVbox.addLayout(canvasHbox, stretch=3)

        ## 整体网格布局
        grid = QtGui.QGridLayout()
        grid.setSpacing(20)
        grid.addLayout(dispHbox,  1, 0)
        grid.addLayout(PanelHbox, 2, 1)
        grid.addLayout(canvasVbox, 2, 0)
        grid.setColumnStretch(0,3)
        grid.setColumnStretch(1,1)

        widget = QtGui.QWidget()
        widget.setLayout(grid)
        self.setCentralWidget(widget)

        ## 槽函数与信号之间进行连接
        self.pointSliderWidget.sliderReleased.connect(self.pointChangedSlot)
        nextButtonWidget.clicked.connect(self.nextButtonClickedSlot)
        prevButtonWidget.clicked.connect(self.prevButtonClickedSlot)
        self.classComboWidget.activated.connect(self.ComboBoxActivatedSlot)
        addButtonWidget.clicked.connect(self.addButtonClickedSlot)
        delButtonWidget.clicked.connect(self.delButtonClickedSlot)
        self.resultTableWidget.cellClicked.connect(self.tableCellClickedSlot)
        self.resultTableWidget.itemClicked.connect(self.tableItemClickedSlot)

    # 快捷键初始化
    def init_Shortcut(self):
        shortcut1 = QtGui.QShortcut(QtGui.QKeySequence("Right"), self)
        shortcut1.activated.connect(self.nextButtonClickedSlot)
        shortcut2 = QtGui.QShortcut(QtGui.QKeySequence("Left"), self)
        shortcut2.activated.connect(self.prevButtonClickedSlot)
        # shortcut3 = QtGui.QShortcut(QtGui.QKeySequence("Enter"), self)
        # shortcut3.activated.connect(self.addButtonClickedSlot)

    #####
    ## Toolbar功能实现函数
    #####

    # 加载文件功能
    def _loadLidarFrame(self):
        fname = QtGui.QFileDialog.getOpenFileName(self, 'Open file','/home/vincentfei')
        self.read_path = os.path.split(fname)[0]
        self.files_list = os.listdir(self.read_path)
        self.files_list.sort()
        self.files_num = len(self.files_list)
        self.current_file = os.path.split(fname)[1]
        self.frame_id = self.files_list.index(self.current_file)

        ## 读取数据
        datamat = mat4py.loadmat(fname)
        self.max_layer_num = datamat["layer_num"]
        self.transform_matrix_local = np.array(datamat["transform_matrix_local"])
        self.transform_matrix_global = np.array(datamat["transform_matrix_global"])
        self.lidar_data = datamat["layer_data"]

        # 旋转变换
        for id in range(0, self.max_layer_num):
            layer_data = np.squeeze(np.array(self.lidar_data[id]))
            position = np.hstack([layer_data[:, 0:3], np.ones([layer_data.shape[0],1])])
            intensity = layer_data[:, 3]
            intensity = np.reshape(intensity, [intensity.shape[0],1])
            transformed = np.dot(self.transform_matrix_local, position.T).T
            self.lidar_data[id] = np.hstack([transformed[:,0:3], intensity])

        ## 帧数改变时内部数据更新
        self.layer_id = 0
        self.point_id = 0
        self.point3D_show = None
        self.pointHei_show = None
        self.pointInt_show = None
        self.current_layer_data = np.squeeze(np.array(self.lidar_data[0]))
        self.max_point_num = self.current_layer_data.shape[0]

        # Table数据刷新
        self.result_table = []
        for id in range(0, self.max_layer_num):
            self.result_table.append([])

        ## 帧数改变时UI刷新
        self._frameUpdate()
        self._pointUpdate()
        self._tableUpdate()

    # 选择帧数功能
    def _selectFrame(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            dlgTitle = "Choose Frame Id"
            question = "Which frame would you like to label ?"
            items =  [ str(i) for i in range(0,self.files_num)]
            (item, ok) = QtGui.QInputDialog.getItem(self, dlgTitle, question, items, 0, False)
            if ok and item:
                self.frame_id = int(item)
                ## 读取数据
                self.current_file = self.files_list[self.frame_id]
                fname = os.path.join(self.read_path, self.current_file)
                datamat = mat4py.loadmat(fname)
                self.max_layer_num = datamat["layer_num"]
                self.transform_matrix_local = np.array(datamat["transform_matrix_local"])
                self.transform_matrix_global = np.array(datamat["transform_matrix_global"])
                self.lidar_data = datamat["layer_data"]

                # 旋转变换
                for id in range(0, self.max_layer_num):
                    layer_data = np.squeeze(np.array(self.lidar_data[id]))
                    position = np.hstack([layer_data[:, 0:3], np.ones([layer_data.shape[0], 1])])
                    intensity = layer_data[:, 3]
                    intensity = np.reshape(intensity, [intensity.shape[0], 1])
                    transformed = np.dot(self.transform_matrix_local, position.T).T
                    self.lidar_data[id] = np.hstack([transformed[:, 0:3], intensity])

                ## 帧数改变时内部数据更新
                self.layer_id = 0
                self.point_id = 0
                self.point3D_show = None
                self.pointHei_show = None
                self.pointInt_show = None
                self.current_layer_data = np.squeeze(np.array(self.lidar_data[0]))
                self.max_point_num = self.current_layer_data.shape[0]
                # Table数据刷新
                self.result_table = []
                for id in range(0, self.max_layer_num):
                    self.result_table.append([])
                ## 帧数改变时UI刷新
                self._frameUpdate()
                self._pointUpdate()
                self._tableUpdate()

    # 后退一帧
    def _backFrame(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.frame_id -= 1
            if self.frame_id < 0:
                self.statusBar().showMessage("Already first frame.")
                self.frame_id += 1
            else:
                ## 读取数据
                self.current_file = self.files_list[self.frame_id]
                fname = os.path.join(self.read_path, self.current_file)
                datamat = mat4py.loadmat(fname)
                self.max_layer_num = datamat["layer_num"]
                self.transform_matrix_local = np.array(datamat["transform_matrix_local"])
                self.transform_matrix_global = np.array(datamat["transform_matrix_global"])
                self.lidar_data = datamat["layer_data"]

                # 旋转变换
                for id in range(0, self.max_layer_num):
                    layer_data = np.squeeze(np.array(self.lidar_data[id]))
                    position = np.hstack([layer_data[:, 0:3], np.ones([layer_data.shape[0], 1])])
                    intensity = layer_data[:, 3]
                    intensity = np.reshape(intensity, [intensity.shape[0], 1])
                    transformed = np.dot(self.transform_matrix_local, position.T).T
                    self.lidar_data[id] = np.hstack([transformed[:, 0:3], intensity])

                ## 帧数改变时内部数据更新
                # self.layer_id = 0
                self.point_id = 0
                self.point3D_show = None
                self.pointHei_show = None
                self.pointInt_show = None
                self.current_layer_data = np.squeeze(np.array(self.lidar_data[self.layer_id]))
                self.max_point_num = self.current_layer_data.shape[0]
                # Table数据刷新
                self.result_table = []
                for id in range(0, self.max_layer_num):
                    self.result_table.append([])
                ## 帧数改变时UI刷新
                self._frameUpdate()
                self._pointUpdate()
                self._tableUpdate()

    # 前进一帧
    def _nextFrame(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.frame_id +=  1
            if self.frame_id >= self.files_num:
                self.statusBar().showMessage("Already last frame.")
                self.frame_id -= 1
            else:
                ## 读取数据
                self.current_file = self.files_list[self.frame_id]
                fname = os.path.join(self.read_path, self.current_file)
                datamat = mat4py.loadmat(fname)
                self.max_layer_num = datamat["layer_num"]
                self.transform_matrix_local = np.array(datamat["transform_matrix_local"])
                self.transform_matrix_global = np.array(datamat["transform_matrix_global"])
                self.lidar_data = datamat["layer_data"]

                # 旋转变换
                for id in range(0, self.max_layer_num):
                    layer_data = np.squeeze(np.array(self.lidar_data[id]))
                    position = np.hstack([layer_data[:, 0:3], np.ones([layer_data.shape[0], 1])])
                    intensity = layer_data[:, 3]
                    intensity = np.reshape(intensity, [intensity.shape[0], 1])
                    transformed = np.dot(self.transform_matrix_local, position.T).T
                    self.lidar_data[id] = np.hstack([transformed[:, 0:3], intensity])

                ## 帧数改变时内部数据更新
                # self.layer_id = 0
                self.point_id = 0
                self.point3D_show = None
                self.pointHei_show = None
                self.pointInt_show = None
                self.current_layer_data = np.squeeze(np.array(self.lidar_data[self.layer_id]))
                self.max_point_num = self.current_layer_data.shape[0]
                # Table数据刷新
                self.result_table = []
                for id in range(0, self.max_layer_num):
                    self.result_table.append([])
                ## 帧数改变时UI刷新
                self._frameUpdate()
                self._pointUpdate()
                self._tableUpdate()


    # 选择线数
    def _selectLayer(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            dlgTitle = "Choose Layer Id"
            question = "Which layer would you like to label ?"
            items =  [ str(i) for i in range(0,self.max_layer_num)]
            (item, ok) = QtGui.QInputDialog.getItem(self, dlgTitle, question, items, 0, False)
            if ok and item:
                ## 线数改变时内部数据更新
                self.layer_id = int(item)
                self.point3D_show = None
                self.pointHei_show = None
                self.pointInt_show = None
                self.current_layer_data = np.squeeze(np.array(self.lidar_data[self.layer_id]))
                self.max_point_num = self.current_layer_data.shape[0]
                ## 线数改变时UI更新
                self._layerUpdate()
                self._pointUpdate()
                self._tableUpdate()

    # 后退一线
    def _backLayer(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.layer_id -= 1
            if self.layer_id < 0:
                self.statusBar().showMessage("Already first layer.")
                self.layer_id += 1
            else:
                ## 线数改变时内部数据更新
                self.point3D_show = None
                self.pointHei_show = None
                self.pointInt_show = None
                self.current_layer_data = np.squeeze(np.array(self.lidar_data[self.layer_id]))
                self.max_point_num = self.current_layer_data.shape[0]
                ## 线数改变时UI更新
                self._layerUpdate()
                self._pointUpdate()
                self._tableUpdate()

    # 前进一线
    def _nextLayer(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.layer_id +=  1
            if self.layer_id >= self.max_layer_num:
                self.statusBar().showMessage("Already last layer.")
                self.layer_id -= 1
            else:
                ## 线数改变时内部数据更新
                self.point3D_show = None
                self.pointHei_show = None
                self.pointInt_show = None
                self.current_layer_data = np.squeeze(np.array(self.lidar_data[self.layer_id]))
                self.max_point_num = self.current_layer_data.shape[0]
                ## 线数改变时UI更新
                self._layerUpdate()
                self._pointUpdate()
                self._tableUpdate()

    # 自动标注
    def _autoLabel(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            modelpath = os.path.join(self.read_path,"Model",str(self.layer_id)+'.pkl')
            if os.path.isfile(modelpath):
                model     = AL.readModel(modelpath)
                filtered  = AL.filterData(self.current_layer_data)
                features  = AL.extractFeatures(self.current_layer_data, filtered)
                predicted = AL.classification(model, features)
                self._autoLabelUpdate(predicted)
            else:
                self.statusBar().showMessage("No model can be found.")
        # 暂时只开发Curb的自动标注功能
        #
        # if self.read_path is None:
        #     self.statusBar().showMessage("Open a folder first.")
        # else:
        #     curbs = []
        #     for layer_id in range(0, self.max_layer_num):
        #         layer_label = self.result_table[layer_id]
        #         i = 0
        #         while i < len(layer_label):
        #             i += 1
        #             if i % 3 == 2:
        #                 if layer_label[i] == 1:
        #                    curb_point = self.lidar_data[layer_id][layer_label[i-1],:]
        #                    curbs.append(curb_point)
        #
        # if len(curbs) >
        pass

    # 储存
    def _save(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            if len(self.result_table[self.layer_id])%3 != 0 == 0:
                box = QtGui.QMessageBox(QtGui.QMessageBox.Warning, "Alert",
                                        "Please complete labeling of current category",
                                        QtGui.QMessageBox.Ok)
                box.exec_()
                return

            count = 0
            for layer in self.result_table:
                if (len(layer) == 0):
                    warning_str = "Layer %d doesn't be labeled !" % (count)
                    reply = QtGui.QMessageBox.question(self, "Message", warning_str,
                                                       QtGui.QMessageBox.Save, QtGui.QMessageBox.Cancel)
                    if reply == QtGui.QMessageBox.Cancel:
                        return
                    elif reply == QtGui.QMessageBox.Save:
                        break
                count += 1

            Label = np.zeros((self.max_layer_num), dtype=np.object)
            for i in range(0, self.max_layer_num):
                layer = self.result_table[i]
                label = -np.ones((self.lidar_data[i].shape[0], 1))
                for j in range(0, len(layer), 3):
                    start = layer[j]
                    end = layer[j+1]
                    category = layer[j+2]
                    for k in range(start,end+1):
                        label[k] = category
                Label[i] = np.array(label)

            datamat = scio.loadmat(os.path.join(self.read_path, self.current_file))
            datamat.update({"label":Label})
            result = datamat

            savepath = os.path.join(self.read_path,"Label",self.current_file)
            scio.savemat(savepath, result)
            message = " %s are successfully saved" % self.current_file
            self.statusBar().showMessage(message)


    #####
    ## 槽函数
    #####

    # 与滑动条相连的槽函数
    def pointChangedSlot(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.point_id = self.pointSliderWidget.value()
            self._pointUpdate()

    # next按钮的槽函数
    def nextButtonClickedSlot(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.point_id += 1
            if self.point_id >= self.max_point_num:
                self.statusBar().showMessage("Already last point.")
                self.point_id -= 1
            else:
                self._pointUpdate()

    # prev按钮的槽函数
    def prevButtonClickedSlot(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            self.point_id -= 1
            if self.point_id < 0:
                self.statusBar().showMessage("Already first point.")
                self.point_id += 1
            else:
                self._pointUpdate()

    # 下拉菜单的槽函数
    def ComboBoxActivatedSlot(self, id):
        # 添加一个功能,也就是没有标注完不能切换类别
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            if (len(self.result_table[self.layer_id])%3 != 0) and (self.category != id):
                box = QtGui.QMessageBox(QtGui.QMessageBox.Warning, "Alert",
                                        "Please complete labeling of current category",
                                        QtGui.QMessageBox.Ok)
                box.exec_()
                self.classComboWidget.setCurrentIndex(self.category)
            else:
                self.category = id

    # Add按钮的槽函数
    def addButtonClickedSlot(self):
        # 不同的标注类别处理方式也不一样
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            if self.category == 0:
                # Road
                if len(self.result_table[self.layer_id]) % 3 == 0:
                    if len(self.result_table[self.layer_id]) != 0:
                        deleted = self.result_table[self.layer_id]
                        for i in range(0, len(self.result_table[self.layer_id]), 3):
                            if self.result_table[self.layer_id][i] == self.point_id:
                                if self.result_table[self.layer_id][i + 2] != self.category:
                                    # 删除前一条记录
                                    del deleted[i + 2]
                                    del deleted[i + 1]
                                    del deleted[i]
                        self.result_table[self.layer_id] = deleted
                    self.result_table[self.layer_id].append(self.point_id)

                else:
                    if self.point_id < self.result_table[self.layer_id][-1]:
                        self.statusBar().showMessage("Illegal add.")
                        return
                    if len(self.result_table[self.layer_id]) // 3 != 0:
                        deleted = self.result_table[self.layer_id]
                        for i in range(0, len(self.result_table[self.layer_id])-1, 3):
                            if self.result_table[self.layer_id][i] == self.point_id:
                                if self.result_table[self.layer_id][i + 2] == self.category:
                                    if self.result_table[self.layer_id][i+1] <= self.point_id:
                                        # 删除前一条记录
                                        del deleted[i + 2]
                                        del deleted[i + 1]
                                        del deleted[i]
                        self.result_table[self.layer_id] = deleted
                    self.result_table[self.layer_id].append(self.point_id)
                    self.result_table[self.layer_id].append(self.category)

                self._tableUpdate()

            elif self.category == 1:
                # Curb
                deleted = self.result_table[self.layer_id]
                for i in range(0, len(self.result_table[self.layer_id]), 3):
                    if self.result_table[self.layer_id][i] == self.point_id:
                        if self.result_table[self.layer_id][i+1] == self.point_id:
                            if self.result_table[self.layer_id][i + 2] == self.category:
                                return
                            else:
                                # 删除前一条记录
                                del deleted[i + 2]
                                del deleted[i + 1]
                                del deleted[i]
                self.result_table[self.layer_id] = deleted

                self.result_table[self.layer_id].append(self.point_id)
                self.result_table[self.layer_id].append(self.point_id)
                self.result_table[self.layer_id].append(self.category)
                self._tableUpdate()

            elif self.category == 2:
                # Noise
                if len(self.result_table[self.layer_id]) % 3 == 0:
                    if len(self.result_table[self.layer_id]) != 0:
                        deleted = self.result_table[self.layer_id]
                        for i in range(0, len(self.result_table[self.layer_id]), 3):
                            if self.result_table[self.layer_id][i] == self.point_id:
                                if self.result_table[self.layer_id][i + 2] != self.category:
                                    # 删除前一条记录
                                    del deleted[i + 2]
                                    del deleted[i + 1]
                                    del deleted[i]
                        self.result_table[self.layer_id] = deleted
                    self.result_table[self.layer_id].append(self.point_id)

                else:
                    if self.point_id < self.result_table[self.layer_id][-1]:
                        self.statusBar().showMessage("Illegal add.")
                        return
                    if len(self.result_table[self.layer_id]) // 3 != 0:
                        deleted = self.result_table[self.layer_id]
                        for i in range(0, len(self.result_table[self.layer_id]) - 1, 3):
                            if self.result_table[self.layer_id][i] == self.point_id:
                                if self.result_table[self.layer_id][i + 2] == self.category:
                                    if self.result_table[self.layer_id][i + 1] <= self.point_id:
                                        # 删除前一条记录
                                        del deleted[i + 2]
                                        del deleted[i + 1]
                                        del deleted[i]
                        self.result_table[self.layer_id] = deleted
                    self.result_table[self.layer_id].append(self.point_id)
                    self.result_table[self.layer_id].append(self.category)

                self._tableUpdate()
            else:
                print("Error!")

    # Del按钮的槽函数
    def delButtonClickedSlot(self):
        if self.read_path is None:
            self.statusBar().showMessage("Open a folder first.")
        else:
            delete_row = self.resultTableWidget.currentRow()
            if delete_row == -1:
                return

            if (delete_row*3+3) > len(self.result_table[self.layer_id]):
                if delete_row*3 < len(self.result_table[self.layer_id]):
                    del self.result_table[self.layer_id][delete_row * 3]
                    self._tableUpdate()
                    return
                else:
                    return

            del self.result_table[self.layer_id][delete_row*3+2]
            del self.result_table[self.layer_id][delete_row*3+1]
            del self.result_table[self.layer_id][delete_row*3]
            self._tableUpdate()


    def tableCellClickedSlot(self,x,y):
        # print(x,',',y)
        # new_item = QtGui.QTableWidgetItem("007")
        # self.resultTableWidget.setItem(x, y, new_item)
        pass

    def tableItemClickedSlot(self,x):
        # print(x)
        pass


    #####
    ## UI界面的更新函数
    #####
    # 帧切换时的刷新函数
    def _frameUpdate(self):
        ## Picture Refresh
        self.axis3D.clear()
        # id1 = 0 if self.layer_id == 0 else (self.layer_id-1)
        # prev_layer_data = np.squeeze(np.array(self.lidar_data[id1]))
        # self.axis3D.plot(prev_layer_data[:, 0], prev_layer_data[:, 1], prev_layer_data[:, 2],
        #                     c='red', marker='.', linewidth=0)

        self.axis3D.plot(self.current_layer_data[:,0], self.current_layer_data[:,1], self.current_layer_data[:,2],
                            c='orange', marker='.', linewidth=0)

        # id2 = (self.max_layer_num - 1) if self.layer_id == (self.max_layer_num - 1) else (self.layer_id + 1)
        # next_layer_data = np.squeeze(np.array(self.lidar_data[id2]))
        # self.axis3D.plot(next_layer_data[:, 0], next_layer_data[:, 1], next_layer_data[:, 2],
        #                     c='black', marker='.', linewidth=0)
        self.canvas3D.draw()

        self.axisHei.clear()
        x = np.array(range(0,self.max_point_num))
        self.axisHei.plot(x, self.current_layer_data[:,2], c='blue', marker='.', linewidth=0)
        self.canvasHei.draw()

        self.axisWin.clear()
        win_start = (self.point_id - self.win_size) if (self.point_id - self.win_size) >= 0 else 0
        win_end = (self.point_id + self.win_size) if (self.point_id - self.win_size) <= (
                    self.max_point_num - 1) else self.max_point_num - 1
        self.axisWin.plot(self.current_layer_data[win_start:self.point_id, 0],
                          self.current_layer_data[win_start:self.point_id, 1],
                          self.current_layer_data[win_start:self.point_id, 2],
                          c='g', marker='.', linewidth=0)
        self.axisWin.plot(self.current_layer_data[self.point_id:win_end, 0],
                          self.current_layer_data[self.point_id:win_end, 1],
                          self.current_layer_data[self.point_id:win_end, 2],
                          c='r', marker='.', linewidth=0)
        self.axisWin.axis('equal')
        self.canvasWin.draw()

        self.axisInt.clear()
        self.axisInt.plot(x, self.current_layer_data[:,3], c='m', marker='.', linewidth=0)
        self.canvasInt.draw()

        ## QSlider Refresh
        self.pointSliderWidget.setMaximum(self.max_point_num)
        self.pointSliderWidget.setValue(0)

        ## QLabel Refresh
        self.frameIdWidget.setText(str(self.frame_id))
        self.layerIdWidget.setText(str(self.layer_id))
        self.pointIdWidget.setText(str(self.point_id))

    # 线切换时的刷新函数
    def _layerUpdate(self):
        ## Picture Refresh
        self.axis3D.clear()
        # id1 = 0 if self.layer_id == 0 else (self.layer_id - 1)
        # prev_layer_data = np.squeeze(np.array(self.lidar_data[id1]))
        # self.axis3D.plot(prev_layer_data[:, 0], prev_layer_data[:, 1], prev_layer_data[:, 2],
        #                     c='red', marker='.', linewidth=0)

        self.axis3D.plot(self.current_layer_data[:, 0], self.current_layer_data[:, 1], self.current_layer_data[:, 2],
                            c='orange', marker='.', linewidth=0)

        # id2 = (self.max_layer_num - 1) if self.layer_id == (self.max_layer_num - 1) else (self.layer_id + 1)
        # next_layer_data = np.squeeze(np.array(self.lidar_data[id2]))
        # self.axis3D.plot(next_layer_data[:, 0], next_layer_data[:, 1], next_layer_data[:, 2],
        #                     c='black', marker='.', linewidth=0)
        self.canvas3D.draw()

        self.axisHei.clear()
        x = np.array(range(0, self.max_point_num))
        self.axisHei.plot(x, self.current_layer_data[:, 2], c='blue', marker='.', linewidth=0)
        self.canvasHei.draw()

        self.axisWin.clear()
        win_start = (self.point_id - self.win_size) if (self.point_id - self.win_size) >= 0 else 0
        win_end = (self.point_id + self.win_size) if (self.point_id - self.win_size) <= (
                    self.max_point_num - 1) else self.max_point_num - 1
        self.axisWin.plot(self.current_layer_data[win_start:self.point_id, 0],
                          self.current_layer_data[win_start:self.point_id, 1],
                          self.current_layer_data[win_start:self.point_id, 2],
                          c='g', marker='.', linewidth=0)
        self.axisWin.plot(self.current_layer_data[self.point_id:win_end, 0],
                          self.current_layer_data[self.point_id:win_end, 1],
                          self.current_layer_data[self.point_id:win_end, 2],
                          c='r', marker='.', linewidth=0)
        self.axisWin.axis('equal')
        self.canvasWin.draw()

        self.axisInt.clear()
        self.axisInt.plot(x, self.current_layer_data[:, 3], c='m', marker='.', linewidth=0)
        self.canvasInt.draw()

        ## QSlider Refresh
        self.pointSliderWidget.setMaximum(self.max_point_num)
        self.pointSliderWidget.setValue(self.point_id)

        ## QLabel Refresh
        self.layerIdWidget.setText(str(self.layer_id))

    # 点切换时的刷新函数
    def _pointUpdate(self):
        ## Picture Refresh
        if self.point3D_show is not None:
            self.axis3D.collections.remove(self.point3D_show)
        self.point3D_show = self.axis3D.scatter(self.current_layer_data[self.point_id, 0],
                                                self.current_layer_data[self.point_id, 1],
                                                self.current_layer_data[self.point_id, 2],
                                                c='b', marker='o', edgecolors='g', linewidths=10)
        self.canvas3D.draw()

        if self.pointHei_show is not None:
            self.axisHei.collections.remove(self.pointHei_show)
        self.pointHei_show = self.axisHei.scatter(self.point_id,
                                                  self.current_layer_data[self.point_id, 2],
                                                  c='r', marker='o', edgecolors='r', linewidths=8)
        self.canvasHei.draw()

        self.axisWin.clear()
        win_start = (self.point_id - self.win_size) if (self.point_id - self.win_size) >= 0 else 0
        win_end = (self.point_id + self.win_size) if (self.point_id - self.win_size) <= (self.max_point_num-1) else self.max_point_num-1
        self.axisWin.plot(self.current_layer_data[win_start:self.point_id, 0],
                          self.current_layer_data[win_start:self.point_id, 1],
                          self.current_layer_data[win_start:self.point_id, 2],
                          c='g', marker='.', linewidth=0)
        self.axisWin.scatter(self.current_layer_data[self.point_id, 0],
                             self.current_layer_data[self.point_id, 1],
                             self.current_layer_data[self.point_id, 2],
                             c='b', marker='o', edgecolors='b', linewidths=5)
        self.axisWin.plot(self.current_layer_data[self.point_id:win_end, 0],
                          self.current_layer_data[self.point_id:win_end, 1],
                          self.current_layer_data[self.point_id:win_end, 2],
                          c='r', marker='.', linewidth=0)
        self.axisWin.axis('equal')
        self.canvasWin.draw()

        if self.pointInt_show is not None:
            self.axisInt.collections.remove(self.pointInt_show)
        self.pointInt_show = self.axisInt.scatter(self.point_id,
                                                  self.current_layer_data[self.point_id, 3],
                                                  c='g', marker='o', edgecolors='g', linewidths=8)
        self.canvasInt.draw()

        ## QLabel Refresh
        self.pointIdWidget.setText(str(self.point_id))

        ## QSlider Refresh
        self.pointSliderWidget.setValue(self.point_id)

    # 表格控件刷新函数
    def _tableUpdate(self):
        self.resultTableWidget.clear()
        for i in range(0, len(self.result_table[self.layer_id])):
            insert_pos = i
            insert_row = insert_pos // 3
            insert_col = insert_pos %  3
            item = QtGui.QTableWidgetItem(str(self.result_table[self.layer_id][insert_pos]))
            self.resultTableWidget.setItem(insert_row, insert_col, item)

    def _autoLabelUpdate(self,predicted):
        for idx in range(predicted.shape[0]):
            if predicted[idx] == 1:
                self.axis3D.scatter(self.current_layer_data[idx, 0],
                                    self.current_layer_data[idx, 1],
                                    self.current_layer_data[idx, 2],
                                    c='b', marker='s', edgecolors='b', linewidths=2)
        self.canvas3D.draw()



# 主函数
def main():
    app = QtGui.QApplication(sys.argv)
    labeltool = LabelTool()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
    pass