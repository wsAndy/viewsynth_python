from __future__ import print_function
import numpy as np
import cv2
import os

cam_data = np.load('./cam_data.npy')
# camName,camPara,p3d,visTable
cam_name = cam_data[0]
cam_param = cam_data[1]
p3d = cam_data[2]
visList = list(cam_data[3])
#[10 12 14 15 17 18 19 20 21 22 23 24 25 28 29]
# print(visList)
cam1_num = 6 # 这个是筛选之后的编号
cam2_num = 4 # visList.index(19)

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
    print('----{0}----'.format(i))
    pos = pos1 + (pos2-pos1)*i/(inter+1)

    tmpom = om*i/(inter+1)

    tmpom = cv2.Rodrigues(tmpom)
    tmpom = tmpom[0]

    target_R = R1.dot(tmpom)
    target_T = -target_R.dot(pos)
    # find nearest camera pos
    # campos : all camera pos
    # pos : this pos
    dis = campos - pos # 3xN
    # dis = np.sqrt(np.sum(dis**2, axis=0)) # 1xN
    # 这边目前其实只要考虑一维度上就可以
    # print(dis.transpose())

    # dis_2 = dis.copy()
    # dis_2 = np.sqrt(np.sum(dis_2**2,axis=0))
    # dis_2_ind = dis_2.argsort()
    # dis_2.sort()

    dis = dis[0]

    left_ind = np.where(dis<0)[0] # 索引,排序前
    right_ind = np.where(dis>=0)[0] # 索引,排序前

    left_dis = np.abs(dis[left_ind])
    right_dis = np.abs(dis[right_ind])

    left_ind_so = left_dis.argsort()
    left_dis.sort()

    right_ind_so = right_dis.argsort()
    right_dis.sort()

    if left_dis.__len__() > left_cam_num:
        # left_cam_num
        left_ind = left_ind[left_ind_so[0:left_cam_num]]
        left_dis = dis[left_ind]
        pass
    else:
        if left_dis.__len__() == 0:
            #
            # left_ind,(array([]),)
            pass
        # left_dis
        else:
            left_ind = left_ind[left_ind_so[0:left_dis.__len__()]]
            left_dis = dis[left_ind]
            pass

    if right_dis.__len__() > right_cam_num:
        # right_cam_num
        right_ind = right_ind[right_ind_so[0:right_cam_num]]
        right_dis = dis[right_ind]
        pass
    else:
        if right_dis.__len__() == 0:
            #[]
            # right_ind,(array([]),)
            pass
        # right_dis
        else:
            right_ind = right_ind[right_ind_so[0:right_dis.__len__()]]
            right_dis = dis[right_ind]
            pass

    # print('----{0}---'.format(i))
    # print('left_ind,{0}'.format(left_ind) )
    # print('left_dis,{0}'.format(left_dis) )
    # print('right_ind,{0}'.format(right_ind) )
    # print('right_dis,{0}'.format(right_dis) )
    # print('----')

    # 把letf与right组合在一起

    dis = np.concatenate([left_dis,right_dis],axis=0)
    ind = np.concatenate([left_ind,right_ind],axis=0)

    dis = np.abs(dis)
    dis_ind = dis.argsort()
    dis_ind = dis_ind[0:use_right_cam_num]
    ind = ind[dis_ind]


    target_pos = [target_R,target_T]
    print(target_pos)
    # print(target_T)
    print(ind)  # 这个是筛选之后的编号
    print('Start to loop warp')
    # 这边甚至可以率先进行一步计算，得到中间结果，比如1号位置使用什么图片编号
    #
    # start to loop warp





