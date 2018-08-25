import os
import numpy as np

import os
import time
import mat4py
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from sklearn import tree
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.externals import joblib


LAYER_INDEX = 3
VALID_LAYER_NUM = 7
WIN_SIZE_ONE_TIMES = 9
WIN_SIZE_TWO_TIMES = 17
WIN_SIZE_FOUR_TIMES = 33


def readModel(path):
    return joblib.load(path)

def filterData(layerdata):

    # 为特征提取做准备
    filtered = np.zeros((layerdata.shape[0], 1))

    for i in range(10, layerdata.shape[0] - 10):
        if layerdata[i, 2] < -0.6:
            # v_left = np.zeros((1,3))
            # v_right = np.zeros((1,3))
            # for j in range(6):
            #     v_left = v_left + layerdata[i+j,0:3] - layerdata[i,0:3]
            #     v_right = v_right + layerdata[i-j,0:3] - layerdata[i,0:3]
            #
            # angle = np.abs(np.dot(v_left, v_right.T) / np.linalg.norm(v_left) / np.linalg.norm(v_right))
            # if angle < 0.866:
            #     filtered[i] = 1
            filtered[i] = 1

    return filtered



def extractFeatures(layerdata,filtered):

    # 为特征提取做准备
    DistanceXY = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的距离
    HeightDiff = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的高度差
    IntensityDiff = np.zeros((layerdata.shape[0], 1))  # 计算相邻两个点间的强度差

    for i in range(layerdata.shape[0] - 1):
        DistanceXY[i] = (layerdata[i + 1, 0] - layerdata[i, 0]) * (layerdata[i + 1, 0] - layerdata[i, 0]) + (
                layerdata[i + 1, 1] - layerdata[i, 1]) * (layerdata[i + 1, 1] - layerdata[i, 1])
        HeightDiff[i] = np.abs(layerdata[i + 1, 2] - layerdata[i, 2])
        IntensityDiff[i] = np.abs(layerdata[i + 1, 3] - layerdata[i, 3]) / 255

    features = np.zeros((layerdata.shape[0], 16))
    start = int((WIN_SIZE_FOUR_TIMES - 1) / 2)
    end = int(layerdata.shape[0] - start)

    for point_index in range(start, end):

        if filtered[point_index] == 0:
            continue

        #####
        # 一倍窗口
        winsize = WIN_SIZE_ONE_TIMES
        winradius = int((winsize - 1) / 2)
        # datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]
        dispack = DistanceXY[point_index - winradius:point_index + winradius, :]
        heipack = HeightDiff[point_index - winradius:point_index + winradius, :]
        intpack = IntensityDiff[point_index - winradius:point_index + winradius, :]

        features[point_index, 0] = np.max(dispack)
        features[point_index, 1] = np.min(dispack)
        features[point_index, 2] = np.mean(dispack)
        features[point_index, 3] = np.var(dispack)

        features[point_index, 4] = np.max(heipack)
        features[point_index, 5] = np.min(heipack)
        features[point_index, 6] = np.mean(heipack)
        features[point_index, 7] = np.var(heipack)

        features[point_index, 8] = np.max(intpack)
        features[point_index, 9] = np.min(intpack)
        features[point_index, 10] = np.mean(intpack)
        features[point_index, 11] = np.var(intpack)

        #####
        # 二倍窗口
        winsize = WIN_SIZE_TWO_TIMES
        winradius = int((winsize - 1) / 2)
        datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]
        center_point = datapack[winradius, 0:3]
        left_point = datapack[0, 0:3]
        right_point = datapack[-1, 0:3]
        left_distance = np.linalg.norm(left_point - center_point)
        right_distance = np.linalg.norm(right_point - center_point)
        point_density = max(left_distance, right_distance)
        two_edge_ratio = max(left_distance, right_distance) / min(left_distance, right_distance)

        features[point_index, 12] = point_density
        features[point_index, 13] = two_edge_ratio

        #####
        # 四倍窗口model
        winsize = WIN_SIZE_FOUR_TIMES
        winradius = int((winsize - 1) / 2)
        datapack = layerdata[point_index - winradius:point_index + winradius + 1, :]

        leftP = datapack[0, 0:3] + datapack[1, 0:3] + datapack[2, 0:3]
        rightP = datapack[winsize - 1, 0:3] + datapack[winsize - 2, 0:3] + datapack[winsize - 3, 0:3]
        centerP = datapack[winradius - 1, 0:3] + datapack[winradius, 0:3] + datapack[winradius + 1, 0:3]
        left_vector = leftP - centerP
        right_vector = rightP - centerP
        # two_edge_angle = np.dot(left_vector,right_vector.T)/np.linalg.norm(left_vector)/np.linalg.norm(right_vector)
        two_edge_angle = np.abs(
            np.dot(left_vector, right_vector.T) / np.linalg.norm(left_vector) / np.linalg.norm(right_vector))

        features[point_index, 15] = two_edge_angle

    return features



def classification(model,features):
    predicted = model.predict(features)
    return predicted