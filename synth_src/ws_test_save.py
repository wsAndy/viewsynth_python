from __future__ import print_function
import numpy as np
import cv2
import os

import sys

cam_data = np.load('./cam_data.npy')
# camName,camPara,p3d,visTable
cam_name = cam_data[0]
cam_param = cam_data[1]
p3d = cam_data[2]
visList = list(cam_data[3])
#[10 12 14 15 17 18 19 20 21 22 23 24 25 28 29]
print(visList)
cam1_num = visList.index(19)  # 现在这个比较其实没有意义，因为真实cam位置顺序与这个3,4是不一样的
cam2_num = visList.index(20)

R1 = cam_param[cam1_num][1]
T1 = cam_param[cam1_num][2]
R2 = cam_param[cam2_num][1]
T2 = cam_param[cam2_num][2]
pos1 = -R1.transpose().dot(T1)
pos2 = -R2.transpose().dot(T2)

om = cv2.Rodrigues(R2.dot(R1.transpose()))
om = om[0]

Ri = cam_param[0][1];
Ti = cam_param[0][2];
campos = (-Ri.transpose().dot(Ti));

for i in range(1,cam_name.__len__(),1):
    Ri = cam_param[i][1]
    Ti = cam_param[i][2]
    campos = np.column_stack( (campos, (-Ri.transpose().dot(Ti))) )

left_cam_num = 2
right_cam_num = 2

use_right_cam_num = 3
if use_right_cam_num > left_cam_num + right_cam_num:
    print('ERROR!use_right_cam_num > left_cam_num + right_cam_num')

OutSet = []

check_dir_list = ['./result/museum',
                  './result/fixmuseum',
                  './result/temp'];
for dir_ch in check_dir_list:
    if not os.path.exists(dir_ch):
        os.makedirs(dir_ch)

inter =10; # 中间有多少个差值位置
frame_position = range(1,inter,1)
# 这个也可以自己去指定位置,只是pos 要改一下为直接指定
for i in frame_position:
    pos = pos1 + (pos2-pos1)*i/(inter+1)
    tmpom = om*i/(inter+1)
    tmpom = cv2.Rodrigues(tmpom)
    tmpom = tmpom[0]
    target_R = R1*tmpom
    target_T = -target_R*pos
    # find nearest camera pos
    # campos : all camera pos
    # pos : this pos
    dis = campos - pos # 3xN
    # dis = np.sqrt(np.sum(dis**2, axis=0)) # 1xN
    # 这边目前其实只要考虑一维度上就可以
    # print(dis.transpose())
    dis = dis[0]
    dis_ind = dis.argsort() # 索引从小到大
    dis.sort() # 从小到大

    print('--{0}---'.format(i))
    # print(dis)
    # print('====')
    left_ind = np.where(dis<0) # 索引
    right_ind = np.where(dis>=0) # 索引
    # print('left_ind')
    # print(left_ind)
    # print('right_ind')
    # print(right_ind)

    if left_cam_num > left_ind[0].__len__():
        if left_ind[0].__len__() == 0:
            left_ind = []
            left_dis = []
        else:
            left_ind = left_ind[0]
            left_dis = dis[left_ind[0]]
    else:
        left_ind = left_ind[0][left_ind[0].__len__()-left_cam_num::]
        left_dis = dis[left_ind] # 可能就变成一个数字

    if right_cam_num > right_ind[0].__len__():
        if right_ind[0].__len__() == 0:
            right_ind = []
            right_dis = []
        else:
            right_ind = right_ind[0]
            right_dis = dis[right_ind[0]]
    else:
        right_ind = right_ind[0][0:right_cam_num]
        right_dis = dis[right_ind]


    if left_ind.__len__() == 0 and right_ind.__len__() != 0:
        # print('>>1')
        ind = np.array(right_ind)
        dis = np.array(right_dis)
    if right_ind.__len__() == 0 and left_ind.__len__() != 0:
        # print('>>2')
        ind = np.array(left_ind)
        dis = np.array(left_dis)
    if left_ind.__len__() != 0 and right_ind.__len__() != 0:
        # print('>>3')
        # print(left_ind)
        # print(right_ind)
        if left_ind.__len__() == 1:
            # right_ind.insert(0,left_ind)
            # ind = np.array(right_ind)
            # right_dis.insert(0,left_dis)
            # dis = np.array(right_dis)
            ind = np.concatenate([left_ind,right_ind],axis=0)
            dis = np.concatenate([[left_dis], right_dis], axis=0)
        elif right_ind.__len__() == 1:
            ind = np.concatenate([left_ind,right_ind],axis=0)
            dis = np.concatenate([left_dis, [right_dis] ], axis=0)
        else:
            ind = np.concatenate([left_ind, right_ind], axis=0)
            dis = np.concatenate([left_dis, right_dis], axis=0)
    if left_ind.__len__() == 0 and right_ind.__len__() == 0:
        # print('>>4')
        pass

    # print('IND {0}'.format(ind) )
    ind = ind.reshape(1,-1)
    dis = dis.reshape(1,-1) # 都变为1行
    dis_abs = np.abs(dis) # 现在用abs
    dis_ind = dis_abs.argsort()
    dis_abs.sort()
    # print(dis_abs)
    # print(dis_ind)
    # print(ind[0][dis_ind])
    # 选择前use_right_cam_num 个
    ind = ind[0][dis_ind] # 变为真实cam的index
    ind = ind[0][0:use_right_cam_num]
    print('image name> ')
    for x in ind:
        print('{0} '.format(visList[x]),end='')
    print()
    # print('image name = {0}'.format(visList[ind]) )





