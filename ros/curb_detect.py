#!/usr/bin/env python3
# ROS Kinetic with Python3

import rospy
from std_msgs.msg import String
# pointcloud
import std_msgs
from sensor_msgs.msg import  PointCloud
from geometry_msgs.msg import Point32

import os
import time
import numpy as np
from sklearn.externals import joblib
import scipy.linalg
import queue
import threading


########################################################################################################################
########################################################################################################################
# Global Variable
# 全局变量
########################################################################################################################
VALID_LAYER_NUM = 7
WIN_SIZE_ONE_TIMES  = [ 9, 9, 9, 9, 9, 9, 9]
WIN_SIZE_TWO_TIMES  = [17,17,17,17,17,17,13]
WIN_SIZE_FOUR_TIMES = [33,33,33,33,33,33,21]
POST_PROCESSING_MODE = 'Parab'

# Parameter used in filter function
# 过滤点云需要用的参数
HEIGHT_MAX_T = -0.6
INSTAN_DIFF_T = 0
HEIGHT_DIFF_MAX_T1 = 0.015
HEIGHT_DIFF_MIN_T2 = 0.01
HEIGHT_DIFF_MAX_T2 = 0.2

########################################################################################################################
########################################################################################################################
# A series of RANSAC function
# RANSAC系列函数
########################################################################################################################
# 直线拟合
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

# 平面拟合
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
            PlaneParam = plane_vec
            CurrentBestNum = InliersNum

    return OutliersIdx, InliersIdx, PlaneParam

# Fit 2-Dimension parabola
# 抛物线拟合,只支持二维
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

        # print("Inl:", InliersNum)

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

    ##
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
            print('( filtering              ) Time cost :' , end2 - end1)
            print('( featureExtraction 2    ) Time cost : ', end3 - end2)
            print('( classification    2    ) Time cost : ', end4 - end3)
            print('( postProcessing         ) Time cost : ', end5 - end4)
            print('( ========Total========= ) Time cost : ', end5 - start)

        return pre_processed_data, curbs, fit_results

    ##
    def preProcessing(self,data):
        # 把每一个Layer的数据分开存放到列表里面
        datas_by_layer = []
        layer_num = 0
        start = 0
        for index in range(data.shape[0]):
            if data[index,4] == layer_num:
                continue
            layer_data = data[start:index, 0:5]
            # 滤波操作
            filtered_data = layer_data
            for i in range(1, layer_data.shape[0]-1):
                filtered_data[i, 0] = (layer_data[i+1, 0] + layer_data[i, 0] + layer_data[i-1, 0])/3
                filtered_data[i, 1] = (layer_data[i+1, 1] + layer_data[i, 1] + layer_data[i-1, 1])/3
                filtered_data[i, 2] = (layer_data[i+1, 2] + layer_data[i, 2] + layer_data[i-1, 2])/3

            datas_by_layer.append(filtered_data[1:-1,:])
            layer_num = data[index,4]
            start = index
        datas_by_layer.append(data[start:index, 0:5])

        return datas_by_layer

    ##
    def filter(self, data):
        filtered_by_layer = []
        cnt = 0
        for layer_index in range(len(data)):
            layerdata = data[layer_index]
            # 为特征提取做准备
            filtered = np.zeros((layerdata.shape[0], 1))
            for i in range(10, layerdata.shape[0] - 10):
                if layerdata[i, 2] < HEIGHT_MAX_T:
                    insdiff = layerdata[i + 5, 3] - layerdata[i - 5, 3]
                    if insdiff != INSTAN_DIFF_T:
                        heidiff1 = np.abs(layerdata[i + 1, 2] - layerdata[i - 1, 2])
                        if heidiff1 < HEIGHT_DIFF_MAX_T1:
                            heidiff2 = np.abs(layerdata[i + 5, 2] - layerdata[i - 5, 2])
                            if heidiff2 > HEIGHT_DIFF_MIN_T2 and heidiff2 < HEIGHT_DIFF_MAX_T2:
                                filtered[i] = 1
                                cnt += 1
            filtered_by_layer.append(filtered)
        print("Curb candidate : ", cnt)
        return filtered_by_layer

    ##
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
            intense_diff = np.zeros((layerdata.shape[0], 1))    # 计算相邻两个点间的强度差

            for i in range(layerdata.shape[0]-1):
                distance_xy[i] = (layerdata[i + 1, 0] - layerdata[i, 0])**2 + (layerdata[i + 1, 1] - layerdata[i, 1])**2
                height_diff[i] = np.abs(layerdata[i + 1, 2] - layerdata[i, 2])
                intense_diff[i] = np.abs(layerdata[i + 1, 3] - layerdata[i, 3])/255


            start = int((WIN_SIZE_FOUR_TIMES[layer_index]-1)/2)
            end = int(layerdata.shape[0] - start)

            winsize1 = WIN_SIZE_ONE_TIMES[layer_index]
            winradius1 = int((winsize1 - 1) / 2)
            winsize2 = WIN_SIZE_TWO_TIMES[layer_index]
            winradius2 = int((winsize2 - 1) / 2)
            winsize4 = WIN_SIZE_FOUR_TIMES[layer_index]
            winradius4 = int((winsize4 - 1) / 2)

            for point_index in range(start,end):

                if layerpredict[point_index] == 0:
                    continue

                features = np.zeros(16)
                #####
                # 一倍窗口
                dispack = distance_xy [ point_index-winradius1:point_index+winradius1, : ]
                heipack = height_diff [ point_index-winradius1:point_index+winradius1, : ]
                intpack = intense_diff[ point_index-winradius1:point_index+winradius1, : ]

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
                datapack = layerdata[point_index - winradius2:point_index + winradius2 + 1, :]
                center_point = datapack[winradius2,0:3]
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
                datapack = layerdata[point_index - winradius4:point_index + winradius4 + 1, :]
                leftP= datapack[0,0:3] + datapack[1,0:3] + datapack[2,0:3]
                rightP = datapack[winsize4-1,0:3] + datapack[winsize4-2,0:3] + datapack[winsize4-3,0:3]
                centerP = datapack[winradius4-1,0:3] + datapack[winradius4,0:3] + datapack[winradius4+1,0:3]
                left_vector = leftP - centerP
                right_vector = rightP - centerP
                two_edge_angle = np.abs(np.dot(left_vector,right_vector.T)/np.linalg.norm(left_vector)/np.linalg.norm(right_vector))

                features[15] = two_edge_angle

                data_sets.append(list(layerdata[point_index,0:4]))
                features_sets.append(list(features))

            data_sets = np.array(data_sets)
            features_sets = np.array(features_sets)

            data_by_layer.append(data_sets)
            features_by_layer.append(features_sets)

        return data_by_layer, features_by_layer


    ##
    def featureExtractionComplexSingleLayer(self,layerdata,layerpredict,layerindex):

        data_sets = []
        features_sets = []

        # 为特征提取做准备
        distance_xy = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的距离
        height_diff = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的高度差
        intense_diff = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的强度差

        for i in range(layerdata.shape[0] - 1):
            distance_xy[i] = (layerdata[i + 1, 0] - layerdata[i, 0]) * (layerdata[i + 1, 0] - layerdata[i, 0]) + (
                    layerdata[i + 1, 1] - layerdata[i, 1]) * (layerdata[i + 1, 1] - layerdata[i, 1])
            height_diff[i] = np.abs(layerdata[i + 1, 2] - layerdata[i, 2])
            intense_diff[i] = np.abs(layerdata[i + 1, 3] - layerdata[i, 3]) / 255

        start = int((WIN_SIZE_FOUR_TIMES[layerindex] - 1) / 2)
        end = int(layerdata.shape[0] - start)

        for point_index in range(start, end):

            if layerpredict[point_index] == 0:
                continue

            features = np.zeros(16)
            #####
            # 一倍窗口
            winsize = WIN_SIZE_ONE_TIMES[layerindex]
            winradius = int((winsize - 1) / 2)
            dispack = distance_xy[point_index - winradius:point_index + winradius, :]
            heipack = height_diff[point_index - winradius:point_index + winradius, :]
            intpack = intense_diff[point_index - winradius:point_index + winradius, :]

            features[0] = np.max(dispack)
            features[1] = np.min(dispack)
            features[2] = np.mean(dispack)
            features[3] = np.var(dispack)


            features[4] = np.max(heipack)
            features[5] = np.min(heipack)
            features[6] = np.mean(heipack)
            features[7] = np.var(heipack)

            features[8] = np.max(intpack)
            features[9] = np.min(intpack)
            features[10] = np.mean(intpack)
            features[11] = np.var(intpack)

            #####
            # 二倍窗口
            winsize = WIN_SIZE_TWO_TIMES[layerindex]
            winradius = int((winsize - 1) / 2)
            datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]
            center_point = datapack[winradius, 0:3]
            left_point = datapack[0, 0:3]
            right_point = datapack[-1, 0:3]
            left_distance = np.linalg.norm(left_point - center_point)
            right_distance = np.linalg.norm(right_point - center_point)

            point_density = max(left_distance, right_distance)
            two_edge_ratio = max(left_distance, right_distance) / min(left_distance, right_distance)

            features[12] = point_density
            features[13] = two_edge_ratio

            #####
            # 四倍窗口
            winsize = WIN_SIZE_FOUR_TIMES[layerindex]
            winradius = int((winsize - 1) / 2)
            datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]

            leftP = datapack[0, 0:3] + datapack[1, 0:3] + datapack[2, 0:3]
            rightP = datapack[winsize - 1, 0:3] + datapack[winsize - 2, 0:3] + datapack[winsize - 3, 0:3]
            centerP = datapack[winradius - 1, 0:3] + datapack[winradius, 0:3] + datapack[winradius + 1, 0:3]
            left_vector = leftP - centerP
            right_vector = rightP - centerP
            two_edge_angle = np.abs(
                np.dot(left_vector, right_vector.T) / np.linalg.norm(left_vector) / np.linalg.norm(right_vector))

            features[15] = two_edge_angle

            data_sets.append(list(layerdata[point_index, 0:4]))
            features_sets.append(list(features))

        data_sets = np.array(data_sets)
        features_sets = np.array(features_sets)

        return data_sets,features_sets


    ##
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
                if features[model_index].shape[0] != 0:
                    model = joblib.load(filepath)
                    predicts = model.predict(features[model_index])
                else:
                    predicts = np.array([])
            else:
                predicts = np.zeros((features[model_index].shape[0],1))
            predicts_by_layer.append(predicts)

        return predicts_by_layer

    ##
    def classificationSingleLayer(self,features,layerindex):

        modelpath = os.path.join('./data/CASIA/', 'Model')
        filepath = os.path.join(modelpath, str(layerindex) + '.pkl')
        if os.path.exists(filepath):
            if features.shape[0] != 0:
                model = joblib.load(filepath)
                predicts = model.predict(features)
            else:
                predicts = np.array([])
        else:
            predicts = np.zeros((features.shape[0], 1))
        return predicts

    ##
    def postProcessing(self,datasets,predictsets,mode='Plane'):

        if mode == 'Plane':
            curbs, fit_results = self.postProcessingPlane(datasets,predictsets)
        elif mode == 'Line':
            curbs, fit_results = self.postProcessingLine(datasets,predictsets)
        elif mode == 'Parab':
            curbs, fit_results = self.postProcessingParabola(datasets, predictsets)
        else:
            print("Post processing mode error")
            return
        return curbs, fit_results

    ##
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

    ##
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

    ##
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
        rotmat = np.squeeze(rotmat)
        planedata = curbs[inidx, 0:3]
        rotplanedata = np.dot(rotmat, planedata.T).T
        outidx1, inidx1, parab1 = ransac_fit_parabola(rotplanedata, n_iteration=50)

        if inidx1 is None:
            return curbs, [planedata]

        if outidx1 is None or outidx1.shape[0]==0:
            return curbs, [planedata[inidx1, :]]

        residualdata = planedata[outidx1, :]
        rotresidualdata = rotplanedata[outidx1, :]
        outidx2, inidx2, parab2 = ransac_fit_parabola(rotresidualdata, n_iteration=25)

        if inidx2 is None:
            return curbs, [planedata[inidx1, :]]

        return curbs, [planedata[inidx1, :], residualdata[inidx2, :]]

    ##
    def postProcessingSingleLayer(self,curbs):


        outliers_idx, inliers_idx, plane_param1 = ransac_fit_plane(curbs)
        if inliers_idx is None :
            inliers = None
            return curbs, inliers

        assert(inliers_idx is not None)

        inliers = [ curbs[inliers_idx,:] ]
        # outliers = curbs[outliers_idx,:]

        return curbs, inliers

    ##
    def postProcessingSingleLayerLine(self,curbs):


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

# 多线程的实现,把每一线的特征提取分类过程都均分在多线上
class PcdListenerMT():

    def __init__(self):
        rospy.init_node('curb_detector', anonymous=True)
        rospy.Subscriber("lidardata", PointCloud, self.pcd_resolve_callback)
        self.pub = rospy.Publisher("curb", PointCloud, queue_size=10)
        self.curb_detector = CurbDetector()

        self.q = queue.Queue()
        self.new_job_id = 0
        self.job_id_lock = threading.Lock()

        self.curb_predicts = []
        for i in range(VALID_LAYER_NUM):
            self.curb_predicts.append([])
        self.curb_predicts_lock = threading.Lock()

        self.threadlists = []
        self.eventlists = []
        for thread_name in range(2):
            t = threading.Thread(target=self.get_features)
            t.start()
            self.threadlists.append(t)

        for i in range(VALID_LAYER_NUM):
            e = threading.Event()
            self.eventlists.append(e)

        rospy.spin()

    def pcd_resolve_callback(self,msg):
        start = time.time()

        points = msg.points
        posexyz= np.zeros((len(points),3))
        for i in range(len(points)):
            posexyz[i,0] = points[i].x
            posexyz[i,1] = points[i].y
            posexyz[i,2] = points[i].z
        intens = np.array(msg.channels[0].values).reshape((-1,1))
        ring   = np.array(msg.channels[1].values).reshape((-1,1))
        datas = np.concatenate((posexyz,intens,ring),axis=1)

        pre_processed_data = self.curb_detector.preProcessing(datas)
        filtered_data = self.curb_detector.filter(pre_processed_data)

        for i in range(VALID_LAYER_NUM):
            with self.job_id_lock:
                self.new_job_id += 1
                current_id = self.new_job_id
            self.q.put([pre_processed_data[i],filtered_data[i]])
            self.eventlists[i].clear()
            rospy.loginfo("Added task %i to the queue", current_id)


        for i in range(VALID_LAYER_NUM):
            self.eventlists[i].wait()

        datas = []
        for i in range(VALID_LAYER_NUM):
            if len(self.curb_predicts[i]) == 0:
                continue
            datas = datas + self.curb_predicts[i]
        datas = np.array(datas)

        curbs, results = self.curb_detector.postProcessingSingleLayer(datas)
        curbpcd = self.get_lidar_pcd(results[0])
        self.pub.publish(curbpcd)

        rospy.loginfo("One frame has been solved")
        end = time.time()
        print(end-start)


    def get_features(self):
        while not rospy.is_shutdown():
            try:
                task = self.q.get(block=False) # block设置成False，那么队列假如是空的就不等待，直接进Empty
                ## 提特征
                layerindex = int(task[0][0,4])
                datas,features = self.curb_detector.featureExtractionComplexSingleLayer(task[0],task[1],layerindex)
                predicts = self.curb_detector.classificationSingleLayer(features,layerindex)

                curbs = []
                for point_index in range(predicts.shape[0]):
                    if predicts[point_index] != 1:
                        continue
                    curbs.append(list(datas[point_index, 0:3]))

                with self.curb_predicts_lock:
                    self.curb_predicts[layerindex] = curbs
                    print("Solve %d", layerindex)
                    self.eventlists[layerindex].set()
            except queue.Empty:
                # print("Empty")
                pass

    def get_lidar_pcd(self,data):

        lidar_pcd = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'lidar'
        lidar_pcd.header = header

        for idx in range(data.shape[0]):
            lidar_pcd.points.append(Point32(data[idx, 0], data[idx, 1], data[idx, 2]))

        return lidar_pcd



class PcdListener():
    def __init__(self):
        rospy.init_node('curb_detector', anonymous=True)
        rospy.Subscriber("lidardata", PointCloud, self.pcd_resolve_callback,queue_size=1,buff_size=52428800)
        self.curbpub = rospy.Publisher("curb", PointCloud, queue_size=10)
        self.oripub  = rospy.Publisher("curbpcd", PointCloud, queue_size=10)
        self.curb_detector = CurbDetector()
        rospy.spin()


    def pcd_resolve_callback(self,msg):
        rospy.loginfo("One frame has been solved")
        start = time.time()
        points = msg.points
        posexyz= np.zeros((len(points),3))
        for i in range(len(points)):
            posexyz[i,0] = points[i].x
            posexyz[i,1] = points[i].y
            posexyz[i,2] = points[i].z
        intens = np.array(msg.channels[0].values).reshape((-1,1))
        ring   = np.array(msg.channels[1].values).reshape((-1,1))
        datas  = np.concatenate((posexyz,intens,ring),axis=1)

        datasets, curbs, results = self.curb_detector.detect(datas)

        if 0:
            curbpcd = self.get_lidar_pcd(curbs)
            print("Curbs1: ", curbs.shape)
            self.curbpub.publish(curbpcd)
        else:
            if results is not None:
                if len(results) == 1:
                    curbpcd = self.get_lidar_pcd(results[0])
                    print("Curbs1: ", results[0].shape)
                    self.curbpub.publish(curbpcd)
                elif len(results) == 2:
                    curbresult = np.concatenate((results[0],results[1]),axis=0)
                    print("Curbs2: ", curbresult.shape)
                    curbpcd = self.get_lidar_pcd(curbresult)
                    self.curbpub.publish(curbpcd)

                else:
                    rospy.loginfo("No curb has been detected")
            else:
                rospy.loginfo("No curb has been detected")

        msg.header.stamp = rospy.Time.now()
        self.oripub.publish(msg)
        end = time.time()
        print(end - start)


    def get_lidar_pcd(self,data):

        lidar_pcd = PointCloud()
        header = std_msgs.msg.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = 'lidar'
        lidar_pcd.header = header

        for idx in range(data.shape[0]):
            lidar_pcd.points.append(Point32(data[idx, 0], data[idx, 1], data[idx, 2]))

        return lidar_pcd



if __name__ == '__main__':
    try:
        listener = PcdListener()
    except rospy.ROSInterruptException:
        pass
