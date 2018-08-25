import os
import time
import mat4py
import numpy as np
import scipy.linalg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.externals import joblib
from imblearn.over_sampling import SMOTE


########################################################################################################################
########################################################################################################################
# 全局变量
########################################################################################################################
POST_PROCESSING_MODE = 'Parab'  # Mode : Plane, Line, Parab
TRAIN_MODE = False
VALID_LAYER_NUM = 7
WIN_SIZE_ONE_TIMES  = [ 9, 9, 9, 9, 9, 9, 9]
WIN_SIZE_TWO_TIMES  = [17,17,17,17,17,17,13]
WIN_SIZE_FOUR_TIMES = [33,33,33,33,33,33,21]

HEIGHT_MAX_T = -0.6
INSTAN_DIFF_T = 0
HEIGHT_DIFF_MAX_T1 = 0.015
HEIGHT_DIFF_MIN_T2 = 0.01
HEIGHT_DIFF_MAX_T2 = 0.2
########################################################################################################################
########################################################################################################################
# 数据读取
########################################################################################################################

# 读取单帧数据
def readSingleFrame(path):

    datamat = mat4py.loadmat(path)

    if 0:
        layer_num = datamat["layer_num"]
        transform_matrix_local = np.array(datamat["transform_matrix_local"])
        transform_matrix_global = np.array(datamat["transform_matrix_global"])

    lidar_data = datamat["layer_data"]

    frame = np.empty((0, 6))
    for layerindex in range(VALID_LAYER_NUM):
        data = np.squeeze(np.array(lidar_data[layerindex]))
        index = layerindex * np.ones((data.shape[0],1))
        layer = np.concatenate((data, index, index), axis=1)
        frame = np.concatenate((frame, layer), axis=0)

    return frame

# 读取训练数据
def readTrainData(path):

    files_list = os.listdir(path)
    files_list.sort()
    multiframe = []

    for fileindex in range(len(files_list)):
        filename = files_list[fileindex]
        datamat = mat4py.loadmat(os.path.join(path,filename))

        if 0:
            layer_num = datamat["layer_num"]
            transform_matrix_local = np.array(datamat["transform_matrix_local"])
            transform_matrix_global = np.array(datamat["transform_matrix_global"])

        labels = datamat['label']
        lidar_data = datamat["layer_data"]

        frame = np.empty((0,6))
        for layerindex in range(VALID_LAYER_NUM):
            data = np.squeeze(np.array(lidar_data[layerindex]))
            label = np.squeeze(np.array(labels[layerindex]))
            label = np.reshape(label,(-1,1))
            index = layerindex*np.ones(label.shape)
            layer = np.concatenate((data,label,index),axis=1)
            frame = np.concatenate((frame, layer), axis=0)

        multiframe.append(frame)

    return multiframe

########################################################################################################################
########################################################################################################################
# RANSAC系列函数
########################################################################################################################

def ransac_fit_line(data, n_iteration = 100, n_inliers = 3, t_distance = 0.05):

    if data.shape[0] == 0:
        return None, None, None, None

    CurrentBestNum = 0
    # Inliers = None
    InliersIdx = None
    # Outliers = None
    OutliersIdx = None
    Point = None
    LineParam = None

    for itr in range(n_iteration):
        pt1 = data[np.random.randint(0,data.shape[0]),:]
        pt2 = data[np.random.randint(0,data.shape[0]),:]
        vec21 = pt2-pt1
        if np.linalg.norm(vec21) < 0.01:
            continue

        # 求点到直线的距离
        Vecd1 = data - pt1
        Vec21 = np.tile(vec21,(data.shape[0],1))
        Distance = np.linalg.norm(np.cross(Vecd1, Vec21) / np.linalg.norm(vec21),axis=1)
        InliersIdxPre = np.where(Distance<t_distance)[0]
        InliersNum = InliersIdxPre.shape[0]
        if InliersNum > n_inliers and InliersNum > CurrentBestNum:
            # print(itr)
            # print(InliersNum)
            InliersIdx = InliersIdxPre
            # Inliers = data[InliersIdx,:]
            OutliersIdx = np.where(Distance >= t_distance)[0]
            # Outliers = data[OutliersIdx,:]
            Point = pt1
            LineParam = vec21
            CurrentBestNum = InliersNum

    return OutliersIdx, InliersIdx, Point, LineParam

def ransac_fit_plane(data, n_iteration = 50, n_inliers = 10, t_distance = 0.02):
    if data.shape[0] == 0:
        return None, None, None, None

    n_min_inliers = 5

    CurrentBestNum = 0
    Inliers = None
    InliersIdx = None
    Outliers = None
    OutliersIdx = None
    PlaneParam = None

    for itr in range(n_iteration):
        pts = data[np.random.randint(0, data.shape[0], n_min_inliers),0:3]
        A = pts
        B = np.ones((pts.shape[0],1))
        plane_vec = scipy.linalg.lstsq(A, B)[0]

        Vecdp = data - pts[0, 0:3]
        Distance = np.abs(np.dot(Vecdp,plane_vec) / np.linalg.norm(plane_vec))

        InliersIdxPre = np.where(Distance<t_distance)[0]
        InliersNum = InliersIdxPre.shape[0]

        if InliersNum > n_inliers and InliersNum > CurrentBestNum:
            # print(itr)
            # print(InliersNum)
            InliersIdx = InliersIdxPre
            # Inliers = data[InliersIdx,:]
            OutliersIdx = np.where(Distance >= t_distance)[0]
            # Outliers = data[OutliersIdx,:]
            d = plane_vec[0,0]*pts[0,0] + plane_vec[1,0]*pts[0,1] + plane_vec[2,0]*pts[0,2]
            PlaneParam = np.array([plane_vec[0,0],plane_vec[1,0],plane_vec[2,0],-d])
            CurrentBestNum = InliersNum

    return OutliersIdx, InliersIdx, PlaneParam


def func(params, x):
    a, b, c = params
    return a * x * x + b * x + c

def error(params, x, y):
    return func(params, x) - y

def ransac_fit_parabola(data, n_iteration = 50, n_inliers = 4, t_distance = 0.1):
    if data.shape[0] == 0:
        return None, None, None, None

    p0 = [10, 10, 10]
    CurrentBestNum = 0
    InliersIdx = None
    OutliersIdx = None
    ParabolaParam = None

    print("Plane: ", data.shape[0])

    for itr in range(n_iteration):

        pts = data[np.random.randint(0, data.shape[0], 3), 0:2]
        param = scipy.optimize.leastsq(error, p0, args=(pts[:, 0], pts[:, 1]))[0]

        InliersNum = 0
        InliersIdxPre = []
        OutliersIdxPre = []
        for idx in range(data.shape[0]):
            if np.abs(error(param, data[idx, 0], data[idx, 1])) < t_distance:
                InliersNum += 1
                InliersIdxPre.append(idx)
            else:
                OutliersIdxPre.append(idx)

        if InliersNum > n_inliers and InliersNum > CurrentBestNum:
            # print("Itr:", itr)
            # print("Num:", InliersNum)
            InliersIdx = np.array(InliersIdxPre)
            OutliersIdx = np.array(OutliersIdxPre)
            ParabolaParam = param
            CurrentBestNum = InliersNum

    return OutliersIdx, InliersIdx, ParabolaParam


########################################################################################################################
########################################################################################################################
# Curb Detection 核心类
## detect 流程
# - 数据预处理          preProcessing()
# - 数据过滤           filter()
# - 第二次特征提取      featureExtractionComplex
# - 第二次分类         classification(mode='complex')
# - 数据后处理         postProcessing()
## train 参数介绍
# - mode : 'simple'  : 训练用于初步筛选Curb点的简单分类器
#          'complex' : 训练用于精细判断的分类器
########################################################################################################################

class CurbDetector():
    def __init__(self):
        pass

    def train(self,mode):
        if mode == 'simple':
            self.trainSimpleModel()
        elif mode == 'complex':
            self.trainComplexModel()
        else:
            print("Training mode error")
            return

    def detect(self,data):

        start = time.time()
        pre_processed_data = self.preProcessing(data)
        end1 = time.time()


        simple_predicts = self.filter(pre_processed_data)
        end2 = time.time()

        curb_candidates, complex_features = self.featureExtractionComplex(pre_processed_data, simple_predicts)
        end3 = time.time()

        complex_predicts = self.classification(complex_features, mode='Complex')
        end4 = time.time()

        curbs, fit_results = self.postProcessing(curb_candidates, complex_predicts, mode=POST_PROCESSING_MODE)
        end5 = time.time()

        if 1:
            print('( preProcessing          ) Time cost : ', end1 - start)
            print('( filter                 ) Time cost : ', end2 - end1)
            print('( featureExtraction      ) Time cost : ', end3 - end2)
            print('( classification         ) Time cost : ', end4 - end3)
            print('( postProcessing         ) Time cost : ', end5 - end4)
            print('( ========Total========= ) Time cost : ', end5 - start)

        return pre_processed_data, curbs, fit_results

    def preProcessing(self,data):
        # 把每一个Layer的数据分开存放到列表里面
        datas_by_layer = []
        layer_num = 0
        start = 0
        for index in range(data.shape[0]):
            if data[index,5] == layer_num:
                continue

            layer_data = data[start:index, 0:5]
            # 滤波操作，但是貌似没有多大效果
            if TRAIN_MODE == 1:
                filtered_data = []
                filtered_win = 1
                for i in range(filtered_win, layer_data.shape[0]-filtered_win):
                    distance_front = np.linalg.norm(layer_data[i+1, 0:3] - layer_data[i, 0:3])
                    if distance_front > 0.2:
                        distance_back = np.linalg.norm(layer_data[i-1, 0:3] - layer_data[i, 0:3])
                        if distance_back > 0.2:
                            continue
                        else:
                            new_point = layer_data[i, 0:3]
                            for j in range(1,filtered_win+1):
                                new_point = layer_data[i+filtered_win, 0:3] + new_point + layer_data[i-filtered_win, 0:3]
                            new_point = list(new_point / (filtered_win*2+1))
                            new_point.append(layer_data[i, 3])
                            new_point.append(layer_data[i, 4])
                            filtered_data.append(new_point)
                    else:
                        new_point = layer_data[i, 0:3]
                        for j in range(1, filtered_win + 1):
                            new_point = layer_data[i + filtered_win, 0:3] + new_point + layer_data[i - filtered_win, 0:3]
                        new_point = list(new_point / (filtered_win*2+1))
                        new_point.append(layer_data[i, 3])
                        new_point.append(layer_data[i, 4])
                        filtered_data.append(new_point)
                layer_data = np.array(filtered_data)

            datas_by_layer.append(layer_data)
            layer_num = data[index,5]
            start = index
        datas_by_layer.append(data[start:index, 0:5])

        return datas_by_layer


    def featureExtractionInTrain(self,data):
        #
        features_by_layer = []

        for layer_index in range(len(data)):

            layerdata = data[layer_index]
            # 为特征提取做准备
            DistanceXY = np.zeros((layerdata.shape[0], 1))     # 计算相邻两个点间的距离
            HeightDiff = np.zeros((layerdata.shape[0], 1))     # 计算相邻两个点间的高度差
            IntensityDiff = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的强度差

            for i in range(layerdata.shape[0]-1):
                DistanceXY[i] = (layerdata[i + 1, 0] - layerdata[i, 0]) * (layerdata[i + 1, 0] - layerdata[i, 0]) + (
                        layerdata[i + 1, 1] - layerdata[i, 1]) * (layerdata[i + 1, 1] - layerdata[i, 1])
                HeightDiff[i] = np.abs(layerdata[i + 1, 2] - layerdata[i, 2])
                IntensityDiff[i] = np.abs(layerdata[i + 1, 3] - layerdata[i, 3])/255

            features = np.zeros((layerdata.shape[0],16))
            start = int((WIN_SIZE_FOUR_TIMES[layer_index]-1)/2)
            end = int(layerdata.shape[0] - start)

            for point_index in range(start,end):

                #####
                # 一倍窗口
                winsize = WIN_SIZE_ONE_TIMES[layer_index]
                winradius = int((winsize-1)/2)
                datapack = layerdata[point_index-winradius:point_index+winradius+1,:]
                dispack = DistanceXY[point_index-winradius:point_index+winradius,:]
                heipack = HeightDiff[point_index-winradius:point_index+winradius,:]
                intpack = IntensityDiff[point_index-winradius:point_index+winradius,:]

                features[point_index,0] = np.max(dispack)
                features[point_index,1] = np.min(dispack)
                features[point_index,2] = np.mean(dispack)
                features[point_index,3] = np.var(dispack)

                features[point_index,4] = np.max(heipack)
                features[point_index,5] = np.min(heipack)
                features[point_index,6] = np.mean(heipack)
                features[point_index,7] = np.var(heipack)

                features[point_index,8] = np.max(intpack)
                features[point_index,9] = np.min(intpack)
                features[point_index,10] = np.mean(intpack)
                features[point_index,11] = np.var(intpack)


                #####
                # 二倍窗口
                winsize = WIN_SIZE_TWO_TIMES[layer_index]
                winradius = int((winsize - 1) / 2)
                datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]
                center_point = datapack[winradius,0:3]
                left_point = datapack[0,0:3]
                right_point = datapack[-1,0:3]
                left_distance = np.linalg.norm(left_point - center_point)
                right_distance = np.linalg.norm(right_point - center_point)
                point_density = max(left_distance, right_distance)
                two_edge_ratio = max(left_distance, right_distance) / min(left_distance, right_distance)

                features[point_index, 12] = point_density
                features[point_index, 13] = two_edge_ratio

                #####
                # 四倍窗口
                winsize = WIN_SIZE_FOUR_TIMES[layer_index]
                winradius = int((winsize - 1) / 2)
                datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]

                leftP= datapack[0,0:3] + datapack[1,0:3] + datapack[2,0:3]
                rightP = datapack[winsize-1,0:3] + datapack[winsize-2,0:3] + datapack[winsize-3,0:3]
                centerP = datapack[winradius-1,0:3] + datapack[winradius,0:3] + datapack[winradius+1,0:3]
                left_vector = leftP - centerP
                right_vector = rightP - centerP
                # two_edge_angle = np.dot(left_vector,right_vector.T)/np.linalg.norm(left_vector)/np.linalg.norm(right_vector)
                two_edge_angle = np.abs(np.dot(left_vector,right_vector.T)/np.linalg.norm(left_vector)/np.linalg.norm(right_vector))

                features[point_index, 15] = two_edge_angle

            features_by_layer.append(features)

        return features_by_layer


    def featureExtractionSimple(self,data):
        #
        features_by_layer = []

        for layer_index in range(len(data)):
            layerdata = data[layer_index]
            # 为特征提取做准备
            features = np.zeros((layerdata.shape[0], 3))

            for i in range(5,layerdata.shape[0]-5):
                # distance_xy
                features[i, 0] = (layerdata[i + 5, 0] - layerdata[i, 0]) * (layerdata[i + 5, 0] - layerdata[i, 0]) \
                              + (layerdata[i + 5, 1] - layerdata[i, 1]) * (layerdata[i + 5, 1] - layerdata[i, 1]) \
                              + (layerdata[i - 5, 0] - layerdata[i, 0]) * (layerdata[i - 5, 0] - layerdata[i, 0]) \
                              + (layerdata[i - 5, 1] - layerdata[i, 1]) * (layerdata[i - 5, 1] - layerdata[i, 1])
                # height_diff
                features[i, 1] = np.abs(layerdata[i + 5, 2] - layerdata[i, 2]) + np.abs(layerdata[i - 5, 2] - layerdata[i, 2])
                # intense_diff
                features[i, 2] = (np.abs(layerdata[i + 1, 3] - layerdata[i, 3]) + np.abs(layerdata[i - 1, 3] - layerdata[i, 3]))/255


            features_by_layer.append(features)

        return features_by_layer


    def featureExtractionComplex(self,data,predict):
        #
        data_by_layer = []
        features_by_layer = []

        for layer_index in range(len(data)):
            data_sets = []
            features_sets = []

            layerdata = data[layer_index]
            layerpredict = predict[layer_index]
            # 为特征提取做准备
            distance_xy = np.zeros((layerdata.shape[0], 1))     # 计算相邻两个点间的距离
            height_diff = np.zeros((layerdata.shape[0], 1))     # 计算相邻两个点间的高度差
            intense_diff = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的强度差

            for i in range(layerdata.shape[0]-1):
                distance_xy[i] = (layerdata[i + 1, 0] - layerdata[i, 0]) * (layerdata[i + 1, 0] - layerdata[i, 0]) + (
                        layerdata[i + 1, 1] - layerdata[i, 1]) * (layerdata[i + 1, 1] - layerdata[i, 1])
                height_diff[i] = np.abs(layerdata[i + 1, 2] - layerdata[i, 2])
                intense_diff[i] = np.abs(layerdata[i + 1, 3] - layerdata[i, 3])/255


            start = int((WIN_SIZE_FOUR_TIMES[layer_index]-1)/2)
            end = int(layerdata.shape[0] - start)

            for point_index in range(start,end):

                if layerpredict[point_index] == 0:
                    continue

                features = np.zeros(16)
                #####
                # 一倍窗口
                winsize = WIN_SIZE_ONE_TIMES[layer_index]
                winradius = int((winsize-1)/2)
                dispack = distance_xy[ point_index-winradius:point_index+winradius, : ]
                heipack = height_diff[ point_index-winradius:point_index+winradius, : ]
                intpack = intense_diff[ point_index-winradius:point_index+winradius, : ]


                features[0]  = np.max (dispack)
                features[1]  = np.min (dispack)
                features[2]  = np.mean(dispack)
                features[3]  = np.var (dispack)

                features[4]  = np.max (heipack)
                features[5]  = np.min (heipack)
                features[6]  = np.mean(heipack)
                features[7]  = np.var (heipack)

                features[8]  = np.max (intpack)
                features[9]  = np.min (intpack)
                features[10] = np.mean(intpack)
                features[11] = np.var (intpack)


                #####
                # 二倍窗口
                winsize = WIN_SIZE_TWO_TIMES[layer_index]
                winradius = int((winsize - 1) / 2)
                datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]
                center_point = datapack[winradius,0:3]
                left_point = datapack[0,0:3]
                right_point = datapack[-1,0:3]
                left_distance = np.linalg.norm(left_point - center_point)
                right_distance = np.linalg.norm(right_point - center_point)

                point_density = max(left_distance, right_distance)
                two_edge_ratio = max(left_distance, right_distance) / min(left_distance, right_distance)


                features[12] = point_density
                features[13] = two_edge_ratio

                #####
                # 四倍窗口
                winsize = WIN_SIZE_FOUR_TIMES[layer_index]
                winradius = int((winsize - 1) / 2)
                datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]

                leftP= datapack[0,0:3] + datapack[1,0:3] + datapack[2,0:3]
                rightP = datapack[winsize-1,0:3] + datapack[winsize-2,0:3] + datapack[winsize-3,0:3]
                centerP = datapack[winradius-1,0:3] + datapack[winradius,0:3] + datapack[winradius+1,0:3]
                left_vector = leftP - centerP
                right_vector = rightP - centerP
                # two_edge_angle = np.dot(left_vector,right_vector.T)/np.linalg.norm(left_vector)/np.linalg.norm(right_vector)
                two_edge_angle = np.abs(np.dot(left_vector,right_vector.T)/np.linalg.norm(left_vector)/np.linalg.norm(right_vector))

                features[15] = two_edge_angle

                data_sets.append(list(layerdata[point_index,0:4]))
                features_sets.append(list(features))

            data_sets = np.array(data_sets)
            features_sets = np.array(features_sets)

            data_by_layer.append(data_sets)
            features_by_layer.append(features_sets)

        return data_by_layer, features_by_layer


    def classification(self,features,mode):
        if mode == 'Simple':
            modelpath = os.path.join('./data/CASIA/', 'ModelSimple')
        elif mode == 'Complex':
            modelpath = os.path.join('./data/CASIA/', 'Model')
        else:
            return None

        predicts_by_layer = []
        for model_index in range(VALID_LAYER_NUM):
            filepath = os.path.join(modelpath,str(model_index)+'.pkl')
            if os.path.exists(filepath):
                model = joblib.load(filepath)
                if features[model_index].shape[0] != 0:
                    model = joblib.load(filepath)
                    predicts = model.predict(features[model_index])
                else:
                    predicts = np.array([])
            else:
                predicts = np.zeros((features[model_index].shape[0],1))
            predicts_by_layer.append(predicts)

        return predicts_by_layer


    def filter(self,data):
        filtered_by_layer = []
        for layer_index in range(len(data)):
            layerdata = data[layer_index]
            filtered = np.zeros((layerdata.shape[0], 1))
            for i in range(10,layerdata.shape[0]-10):
                if layerdata[i,2] < HEIGHT_MAX_T:
                    insdiff = layerdata[i + 5, 3] - layerdata[i - 5, 3]
                    if insdiff != INSTAN_DIFF_T:
                        heidiff1 = np.abs(layerdata[i + 1, 2] - layerdata[i - 1, 2])
                        if heidiff1 < HEIGHT_DIFF_MAX_T1:
                            heidiff2 = np.abs(layerdata[i + 5, 2] - layerdata[i - 5, 2])
                            if heidiff2 > HEIGHT_DIFF_MIN_T2 and heidiff2 < HEIGHT_DIFF_MAX_T2:
                                filtered[i] = 1
            filtered_by_layer.append(filtered)
        return filtered_by_layer



    def postProcessing(self,datasets,predictsets,mode='Plane'):

        if mode == 'Plane':
            curbs, fit_results = self.postProcessingPlane(datasets,predictsets)
        elif mode == 'Line':
            curbs, fit_results = self.postProcessingLine(datasets,predictsets)
        elif mode == 'Parab':
            curbs, fit_results = self.postProcessingParabola(datasets,predictsets)
        else:
            print("Post processing mode error")
            return
        return curbs, fit_results


    def postProcessingPlane(self,datasets,predictsets):

        curbs = []
        for layer_index in range(len(predictsets)):
            datas = datasets[layer_index]
            predicts = predictsets[layer_index]
            for point_index in range(predicts.shape[0]):
                if predicts[point_index] != 1:
                    continue
                curbs.append(list(datas[point_index, 0:3]))
        curbs = np.array(curbs)

        outliers_idx, inliers_idx, plane_param1 = ransac_fit_plane(curbs)
        if inliers_idx is None :
            inliers = None
            return curbs, inliers

        assert(inliers_idx is not None)

        inliers = [ curbs[inliers_idx,:] ]
        # outliers = curbs[outliers_idx,:]

        return curbs, inliers


    def postProcessingLine(self,datasets,predictsets):

        curbs = []
        for layer_index in range(len(predictsets)):
            datas = datasets[layer_index]
            predicts = predictsets[layer_index]
            for point_index in range(predicts.shape[0]):
                if predicts[point_index] != 1:
                    continue
                curbs.append(list(datas[point_index, 0:3]))
        curbs = np.array(curbs)

        outliers_idx1, inliers_idx1, point1, line_param1 = ransac_fit_line(curbs, n_iteration=50)
        if inliers_idx1 is None or outliers_idx1 is None:
            inliers = None
            return curbs, inliers

        assert(inliers_idx1 is not None)
        assert(outliers_idx1 is not None)

        inliers1 = curbs[inliers_idx1,:]
        outliers1 = curbs[outliers_idx1,:]

        outliers_idx2, inliers_idx2, point2, line_param2 = ransac_fit_line(outliers1, n_iteration=25)
        if inliers_idx2 is None:
            inliers = [inliers1]
            return curbs, inliers
        inliers2 = outliers1[inliers_idx2, :]
        # outliers2 = outliers1[outliers_idx2, :]
        inliers = [inliers1, inliers2]

        return curbs, inliers


    def postProcessingParabola(self,datasets,predictsets):
        curbs = []
        for layer_index in range(len(predictsets)):
            datas = datasets[layer_index]
            predicts = predictsets[layer_index]
            for point_index in range(predicts.shape[0]):
                if predicts[point_index] != 1:
                    continue
                curbs.append(list(datas[point_index, 0:3]))
        curbs = np.array(curbs)

        if curbs.shape[0] == 0:
            return None, None

        outidx, inidx, plane = ransac_fit_plane(curbs)

        if inidx is None:
            return curbs, None

        cost = plane[2] / np.linalg.norm(plane[0:3])
        sint = np.sqrt(1 - cost ** 2)
        rotvec = np.array([plane[1], -plane[0], 0]) / np.sqrt(plane[0] ** 2 + plane[1] ** 2)
        rotmat = np.array([[rotvec[0] ** 2 * (1 - cost) + cost, rotvec[0] * rotvec[1] * (1 - cost) - rotvec[2] * sint,
                            rotvec[0] * rotvec[2] * (1 - cost) + rotvec[1] * sint],
                           [rotvec[0] * rotvec[1] * (1 - cost) + rotvec[2] * sint, rotvec[1] ** 2 * (1 - cost) + cost,
                            rotvec[1] * rotvec[2] * (1 - cost) - rotvec[0] * sint],
                           [rotvec[0] * rotvec[2] * (1 - cost) - rotvec[1] * sint,
                            rotvec[1] * rotvec[2] * (1 - cost) + rotvec[0] * sint, rotvec[2] ** 2 * (1 - cost) + cost]])

        planedata = curbs[inidx, 0:3]
        rotplanedata = np.dot(rotmat, planedata.T).T
        outidx1, inidx1, parab1 = ransac_fit_parabola(rotplanedata)

        if inidx1 is None:
            return curbs, [planedata]

        if outidx1 is None or outidx1.shape[0] == 0:
            return curbs, [planedata[inidx1, :]]

        residualdata = planedata[outidx1, :]
        rotresidualdata = rotplanedata[outidx1, :]
        outidx2, inidx2, parab2 = ransac_fit_parabola(rotresidualdata)

        if inidx2 is None:
            return curbs, [planedata[inidx1, :]]

        return curbs, [planedata[inidx1, :], residualdata[inidx2, :]]


    def trainSimpleModel(self):
        path = '/home/vincentfei/PaparWorkspace/CASIA/Label'
        trainDataset = readTrainData(path)
        allfeatures = {}

        for frame_index in range(len(trainDataset)):
            print(frame_index)
            oneframedata = trainDataset[frame_index]
            oneframedatas = self.preProcessing(oneframedata)
            oneframefeatures = self.featureExtractionSimple(oneframedatas)

            for layer_index in range(len(oneframedatas)):
                if allfeatures.get(layer_index) is None:
                    allfeatures.update({layer_index:[]})
                onelayerdatas = oneframedatas[layer_index]
                onelayerfeatures = oneframefeatures[layer_index]
                Samples = allfeatures[layer_index]
                for pointindex in range(onelayerdatas.shape[0]):
                    if onelayerdatas[pointindex,4] == 1:
                        sample = list(np.hstack([onelayerfeatures[pointindex,:],1]))
                        Samples.append(sample)
                    if np.random.uniform(0,1,1) < 0.05:
                        sample = list(np.hstack([onelayerfeatures[pointindex, :], 0]))
                        Samples.append(sample)
                allfeatures.update({layer_index:Samples})

        for layer_index in range(len(allfeatures)):
            dataset = np.array(allfeatures[layer_index])
            input = dataset[:,0:(dataset.shape[1]-1)]
            output = dataset[:,(dataset.shape[1]-1)]

            ## 划分训练数据与测试数据
            train_data, test_data, train_label, test_label = train_test_split(input, output, test_size=.1,
                                                                              random_state=0)

            train_label = np.reshape(train_label, (-1))

            nocurb_num = np.sum(train_label == 0)
            curb_num = np.sum(train_label == 1)

            if curb_num == 0:
                continue

            ## 训练模型
            # 决策树
            # 设置分类器参数
            clf = tree.DecisionTreeClassifier()
            clf.set_params(class_weight={0: nocurb_num, 1: curb_num})
            # clf.set_params(max_features=5)
            print("Layer num : ", layer_index)
            print(clf)
            # 交叉验证的得分
            scores = cross_val_score(clf, train_data, train_label, cv=5)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            # 训练
            time_fit_start = time.time()
            clf.fit(train_data, train_label)
            time_fit_end = time.time()
            # 验证集上的预测
            time_pre_start = time.time()
            predicted = clf.predict(test_data)
            time_pre_end = time.time()

            print("Tsetsets accuarcy: ", metrics.accuracy_score(test_label, predicted))
            print("Confusion matrix: \n", metrics.confusion_matrix(test_label, predicted))
            print("Time Fit: ", time_fit_end - time_fit_start)
            print("Time Pre: ", time_pre_end - time_pre_start)

            savepath = os.path.join('/home/vincentfei/PaparWorkspace/CASIA/','ModelSimple',str(layer_index)+'.pkl')
            joblib.dump(clf, savepath)

    def trainComplexModel(self):

        path = './data/CASIA/Label'
        trainDataset = readTrainData(path)

        allfeatures = {}

        for frame_index in range(len(trainDataset)):
            print("Training Frame : ",frame_index)
            oneframedata = trainDataset[frame_index]
            oneframedatas = self.preProcessing(oneframedata)
            oneframefeatures = self.featureExtractionInTrain(oneframedatas)

            noise_intensity = 0.1*np.array([1e-4, 1e-5, 1e-4, 1e-8, 1e-3, 1e-5, 1e-4, 1e-6, 1e-3, 1e-5, 1e-4, 1e-6, 1e-2, 1e-1, 0, 1e-2])
            np.random.randn()
            for layer_index in range(len(oneframedatas)):
                if allfeatures.get(layer_index) is None:
                    allfeatures.update({layer_index: []})
                onelayerdatas = oneframedatas[layer_index]
                onelayerfeatures = oneframefeatures[layer_index]
                Samples = allfeatures[layer_index]
                for pointindex in range(onelayerdatas.shape[0]):
                    if onelayerdatas[pointindex, 4] == 1:
                        sample = list(np.hstack([onelayerfeatures[pointindex, :], 1]))
                        Samples.append(sample)

                        for i in range(1):
                            sample_noise = ( onelayerfeatures[pointindex, :] + np.random.randn(1,16)*noise_intensity ).tolist()[0]
                            sample_noise.append(1)
                            Samples.append(sample_noise)

                    if np.random.uniform(0, 1, 1) < 0.05:
                        sample = list(np.hstack([onelayerfeatures[pointindex, :], 0]))
                        Samples.append(sample)
                allfeatures.update({layer_index: Samples})

        for layer_index in range(len(allfeatures)):

            if layer_index != 3:
                continue

            dataset = np.array(allfeatures[layer_index])

            input = dataset[:, 0:(dataset.shape[1] - 1)]
            output = dataset[:, (dataset.shape[1] - 1)]

            ## 划分训练数据与测试数据
            train_data, test_data, train_label, test_label = train_test_split(input, output, test_size=.1,
                                                                              random_state=0)

            train_data_smote, train_label_smote = SMOTE(random_state=42).fit_sample(train_data, train_label)
            train_data = train_data_smote
            train_label = train_label_smote

            train_label = np.reshape(train_label, (-1))

            nocurb_num = np.sum(train_label == 0)
            curb_num = np.sum(train_label == 1)

            if curb_num == 0:
                continue

            ## 训练模型
            # Random Forest Model
            # 比较了各种类型的SVM和RF模型后，发现还是RF模型更适合于这个任务
            # 设置分类器参数
            clf = ensemble.RandomForestClassifier(max_features="auto", random_state=10)
            clf.set_params(n_estimators=10)
            clf.set_params(max_features='log2')
            clf.set_params(class_weight={0: nocurb_num, 1: curb_num})
            # clf.set_params(max_features=5)
            print("Layer num : ", layer_index)
            print(clf)
            # 交叉验证的得分
            scores = cross_val_score(clf, train_data, train_label, cv=5)
            print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            # 训练
            time_fit_start = time.time()
            clf.fit(train_data, train_label)
            time_fit_end = time.time()
            # 验证集上的预测
            time_pre_start = time.time()
            predicted = clf.predict(test_data)
            time_pre_end = time.time()

            print("Tsetsets accuarcy: ", metrics.accuracy_score(test_label, predicted))
            print("Confusion matrix: \n", metrics.confusion_matrix(test_label, predicted))
            print("Time Fit: ", time_fit_end - time_fit_start)
            print("Time Pre: ", time_pre_end - time_pre_start)

            savepath = os.path.join('./data/CASIA/', 'Model', str(layer_index) + '.pkl')
            joblib.dump(clf, savepath)


    # def visualization_clf(self):
    #     modelpath = os.path.join('/home/vincentfei/PaparWorkspace/CASIA/', 'Model')
    #     models_list = os.listdir(modelpath)
    #     models_list.sort()
    #     predicts_by_layer = []
    #     for model_index in range(7):
    #         path = os.path.join(modelpath, models_list[model_index])
    #         model = joblib.load(path)
    #
    #         if 1:
    #             feature_names = ["max_interval", "min_interval", "mean_interval", "var_interval", \
    #                              "max_heightdiff", "min_heightdiff", "mean_heightdiff", "var_heightdiff", \
    #                              "max_intensitydiff", "min_intensitydiff", "mean_intensitydiff", "var_intensitydiff", \
    #                              "point_density", "two_edge_ratio", "Zero", "two_edge_angle"]
    #
    #             # Estimators = model.estimators_
    #             # for index, model in enumerate(Estimators):
    #             #     filename = 'visualization/tree_' + str(index) + '.pdf'
    #             #     dot_data = tree.export_graphviz(model, out_file=None,
    #             #                                     feature_names=feature_names,
    #             #                                     class_names=["road","curb","noise"],
    #             #                                     filled=True, rounded=True,
    #             #                                     special_characters=True)
    #             #     graph = pydotplus.graph_from_dot_data(dot_data)
    #             #     graph.write_pdf(filename)
    #
    #             y_importances = model.feature_importances_
    #             x_importances = feature_names
    #             y_pos = np.arange(len(x_importances))
    #             # 横向柱状图
    #             plt.barh(y_pos, y_importances, align='center')
    #             plt.yticks(y_pos, x_importances)
    #             plt.xlabel('Importances')
    #             plt.xlim(0, 1)
    #             plt.title('Features Importances'# if fit_results is not None:
    # #     # fig = plt.figure("Visualization")
    # #     # ax = Axes3D(fig)
    # #     ax.plot(curbs[:, 0], curbs[:, 1], curbs[:, 2], c='g', marker='.', linewidth=0)
    # #     ax.plot(fit_results[0][:, 0], fit_results[0][:, 1], fit_results[0][:, 2], c='r', marker='.', linewidth=0)
    # #     if len(fit_results) > 1:/home/vincentfei/PaparWorkspace/algor
    # #         ax.plot(fit_results[1][:, 0], fit_results[1][:, 1], fit_results[1][:, 2], c='g', marker='.', linewidth=0)
    # #     plt.show())
    #             plt.show()
    #     return predicts_by_layer




def test():
    dirpath = './data/CASIA/'
    files_list = os.listdir(dirpath)
    files_list.sort()
    files_list = files_list [200:]
    detector = CurbDetector()
    count = 0
    for filename in files_list:
        count += 1
        if count%10 != 0:
            continue
        filepath = dirpath + filename
        print("=============================================================")
        print("File path : ", filepath)
        data = readSingleFrame(filepath)
        datasets, curbs,results = detector.detect(data)
        print("Count  : ", count)
        print("Curbs  : ", curbs.shape[0])
        if results is not None:
            print("Edge 1 : ", results[0].shape[0])
            if len(results) > 1:
                print("Edge 2 : ", results[1].shape[0])
                print("ResNum : ", curbs.shape[0] - results[0].shape[0] - results[1].shape[0] )

        fig = plt.figure("Visualization")
        ax = Axes3D(fig)
        for layer_index in range(len(datasets)):
            layerdata = datasets[layer_index]
            ax.plot(layerdata[:, 0], layerdata[:, 1], layerdata[:, 2], c='y', marker='.',linewidth=0)

        ax.plot(curbs[:, 0], curbs[:, 1], curbs[:, 2], c='g', marker='.', linewidth=0)
        if results is not None:
            ax.plot(results[0][:, 0], results[0][:, 1], results[0][:, 2], c='r', marker='.', linewidth=0)
            if len(results) > 1:
                ax.plot(results[1][:, 0], results[1][:, 1], results[1][:, 2], c='m', marker='.', linewidth=0)
        plt.ion()
        plt.pause(2)  # 显示秒数
        plt.close()



def main():
    data = readSingleFrame('./data/CASIA/1526096356486947_ring_9_0.000000_0.000000_3.138964.mat')
    
    detector = CurbDetector()


    pre_processed_data = detector.preProcessing(data)
    simple_predicts = detector.filter(pre_processed_data)
    curb_candidates, complex_features = detector.featureExtractionComplex(pre_processed_data, simple_predicts)

    count = 0
    fig = plt.figure("Visualization")
    ax = Axes3D(fig)

    for i in range(VALID_LAYER_NUM):
        layerdata = pre_processed_data[i]
        ax.plot(layerdata[:, 0], layerdata[:, 1], layerdata[:, 2], c='y', marker='.', linewidth=0)
        curbs = curb_candidates[i]
        if curbs.shape[0] != 0:
            ax.plot(curbs[:, 0], curbs[:, 1], curbs[:, 2], c='g', marker='.', linewidth=0)
            count = count + curbs.shape[0]
    print(count)

    complex_predicts = detector.classification(complex_features, mode='Complex')
    end5 = time.time()

    curbs, fit_results = detector.postProcessing(curb_candidates, complex_predicts, mode=POST_PROCESSING_MODE)
    end6 = time.time()

    print(curbs.shape)
    print(end6-end5)
    if fit_results is not None:
        ax.plot(curbs[:, 0], curbs[:, 1], curbs[:, 2], c='r', marker='.', linewidth=0)
        ax.plot(fit_results[0][:, 0], fit_results[0][:, 1], fit_results[0][:, 2], c='b', marker='.', linewidth=0)
        if len(fit_results) > 1:
            ax.plot(fit_results[1][:, 0], fit_results[1][:, 1], fit_results[1][:, 2], c='g', marker='.', linewidth=0)

    plt.show()




#
if __name__ == '__main__':

    if TRAIN_MODE:
        detector = CurbDetector()
        detector.train('complex')

    main()
