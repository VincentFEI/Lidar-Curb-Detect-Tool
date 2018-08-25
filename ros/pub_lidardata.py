#!/usr/bin/env python3
# ROS Kinetic with Python3

import rospy
# pointcloud
import std_msgs
from sensor_msgs.msg import  PointCloud
from geometry_msgs.msg import Point32
from sensor_msgs.msg import ChannelFloat32
# transform
import tf

import os
import mat4py
import numpy as np

VALID_LAYER_NUM = 7


def readSingleFrame(path):

    datamat = mat4py.loadmat(path)

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

    return frame,transform_matrix_global


def get_lidar_pcd(data):
    lidar_pcd = PointCloud()
    header = std_msgs.msg.Header()
    header.stamp = rospy.Time.now()
    header.frame_id = 'lidar'
    lidar_pcd.header = header
    channel1 = ChannelFloat32()
    channel1.name = "intensity"
    channel2 = ChannelFloat32()
    channel2.name = "ring"

    for idx in range(data.shape[0]):
        lidar_pcd.points.append(Point32(data[idx,0],data[idx,1],data[idx,2]))
        channel1.values.append(data[idx,3])
        channel2.values.append(data[idx,4])

    lidar_pcd.channels.append(channel1)
    lidar_pcd.channels.append(channel2)

    return lidar_pcd


def talker():
    dirpath = './data/CASIA/'
    files_list = os.listdir(dirpath)
    files_list.sort()
    files_list = files_list[200:]

    tfpub = tf.TransformBroadcaster()

    pcdpub = rospy.Publisher("lidardata", PointCloud, queue_size=1)
    rospy.init_node('lidardataPuber', anonymous=True)
    rate = rospy.Rate(4)

    frame_cnt = 0
    while not rospy.is_shutdown():

        print(frame_cnt)
        filepath = dirpath + files_list[frame_cnt]
        lidardata,transforms = readSingleFrame(filepath)
        ladar_pcd = get_lidar_pcd(lidardata)
        frame_cnt += 1
        tfpub.sendTransform((0,0,0),
                            #(transforms[0,3],transforms[1,3],transforms[2,3]),
                            tf.transformations.quaternion_from_matrix(transforms),
                            rospy.Time.now(),
                            "lidar",
                            "map")
        rospy.loginfo("Lidar Message has been pubbed")
        pcdpub.publish(ladar_pcd)
        rate.sleep()


if __name__ == '__main__':
    try:
        talker()
    except rospy.ROSInterruptException:
        pass


