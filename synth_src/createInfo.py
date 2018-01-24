import multiprocessing
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import slic
import numpy as np
import time
import cv2

cam_data = np.load('./cam_data.npy')
# camName,camPara,p3d,visSet,visTable
cam_name = cam_data[0]
cam_para = cam_data[1]
p3d = cam_data[2]
visSet = cam_data[3]
visTable = cam_data[4]

tmp = io.imread(cam_name[0][0])
rows = tmp.shape[0]
cols = tmp.shape[1]



def saveSpflag(slic_label_name, dmap_name, saveindex):
    label = np.load(slic_label_name)
    dmap = np.load(dmap_name)
    # dmap_x = dmap[:,1]
    # dmap_y = dmap[:,2]
    # dmap_d = dmap[:,3]

    Nsp = label.max()+1 #0--max, num=max+1
    Pcnt = label.sum()
    th = 0.1*Pcnt/Nsp
    Spflag = np.zeros([Nsp,1])

    for i in range(0,Nsp):
        # find region whose ID=i
        # judge id the dmap in this region has enough depth point
        # true: set spflag[i] = 1
        # False: spflag[i] = 0
        sp_ind = np.where(label == i)
        dep_point_count = 0
        for dep_ind in range(0,sp_ind[0].shape[0]):
            # pixel( sp_ind[dep] )
            sp_x = int(round(sp_ind[0][dep_ind]))
            sp_y = int(round(sp_ind[1][dep_ind]))
            # judge if enough depth point occurs in this region
            if abs(dmap[sp_x, sp_y]) > 0.1:
                dep_point_count = dep_point_count + 1
        if dep_point_count < th:
            Spflag[i] = 1

    np.save('./depth/spflag' + str(saveindex) + '.npy', Spflag)


# 构建一个索引
def saveDepth(p3d, pos, rows, cols):
    print('saveDepth')
    p3d_bool = np.sum(p3d,axis=0)
    p3d_bool = p3d_bool.reshape(1,-1)
    p3d_use_pix_ind = np.where(p3d_bool !=0 )
    first_ind = p3d_use_pix_ind[1][0]
    p3d_ = p3d[:,first_ind]
    p3d_ = p3d_.reshape(3,1)
    for x in range(1,p3d_use_pix_ind[1].shape[0]):
        ind = p3d_use_pix_ind[1][x]
        p3d_tmp = p3d[:,ind]
        p3d_tmp = p3d_tmp.reshape(3,1)
        p3d_ = np.concatenate([p3d_,p3d_tmp],axis=1)
    p3d = p3d_
    # need project 3d to 2d
    # To Do
    Q = pos[0]
    R = pos[1]
    T = pos[2]
    M = np.concatenate([R,T],axis=1)
    P_rot = M.dot(np.concatenate([p3d, np.ones([1,p3d.shape[1]] )],axis=0))  # p3d一开始产生就错了
    proj_x = Q[1]+Q[0]*(P_rot[0,:]/P_rot[2,:])
    proj_y = Q[2]+Q[0]*(P_rot[1,:]/P_rot[2,:])
    proj_x = proj_x.reshape(1, -1)
    proj_y = proj_y.reshape(1, -1)
    proj_point = np.concatenate([proj_x,proj_y],axis=0)
    proj_point_dep = P_rot[2,:]
    proj_point_dep = proj_point_dep.reshape(1,-1)
    # depthArray = []
    # 创建一个image大小的矩阵
    mapD = np.zeros([rows, cols])
    for x in range(0,proj_point_dep.shape[1]):
        p_x = int(round(proj_point[0][x]))
        p_y = int(round(proj_point[1][x]))
        if p_x < 0 or p_x >= cols or p_y < 0 or p_y >= rows:
            continue
        # tmp = [p_x,p_y,proj_point_dep[0][x]]
        # print('x='+str(x) )
        # print('proj_point_dep[0] shape={0}'.format(proj_point_dep[0].shape))
        # print('p_x = {0}, p_y = {1}, mapD shape={2}'.format(p_x,p_y,mapD.shape))
        mapD[p_y,p_x] = proj_point_dep[0][x]
        # depthArray.append(tmp)
    # depthArray = np.array(depthArray,np.dtype=float16) # size x 3
    # return depthArray
    # print('leave')
    return mapD



def saveSlic(x):
    print('Start img={0}'.format(x))
    time1 = time.time()
    img = img_as_float(io.imread(cam_name[x][0]))
    seg = slic(img, n_segments= 400, sigma=3)

    # 保存为npy之后，可以得到seg的label，每一个位置的value为超像素ID
    np.save('./slic/seg'+str(x)+'.npy',seg)
    # 直接存储没有意义，因为这边使用像素值表示superpixel的ID
    # io.imsave('./slic/seg' + str(x) + '.png', seg)
    # cv2.imwrite('./slic/segcv2_'+str(x)+'.png',seg)
    print('End: {0}'.format(time.time() - time1))



def createDepth(i):
    pos = cam_para[visSet.index(i)]
    visBool = visTable[:, i] > 0
    onecam_p3d = visBool.transpose() * p3d
    # onecam_p3d: cam id=i, see the p3d
    # depthArray = saveDepth(onecam_p3d, pos, rows, cols)
    mapD = saveDepth(onecam_p3d, pos, rows, cols)
    print(mapD.shape)
    print('---{0}'.format(i))
    np.save('./depth/dep'+str(i)+'.npy',mapD)



def createSpflag(slic_label_name, dmap_name, i):
    saveSpflag(slic_label_name, dmap_name, i)
    print('==')



if __name__ == '__main__':

    USE_POOL = False

    if USE_POOL:
        pool = multiprocessing.Pool(processes=8)
        for i in range(0,cam_name.shape[0]):
            pool.apply_async(saveSlic, (i,))
        pool.close()
        pool.join()



    pool = multiprocessing.Pool(processes=4)
    for i in range(0, visSet.__len__()):
        pool.apply_async(createDepth, (visSet[i],))
    pool.close()
    pool.join()
    # for i in range(0, visSet.__len__()):
    #     createDepth(visSet[i])

    pool = multiprocessing.Pool(processes=4)
    for i in range(0, cam_name.__len__()):
        pool.apply_async(createSpflag, ('./slic/seg'+str(i)+'.npy', './depth/dep'+str(i)+'.npy', i,))
    pool.close()
    pool.join()

    # for i in range(0, cam_name.__len__()):
    #     createSpflag('./slic/seg'+str(i)+'.npy', './depth/dep'+str(i)+'.npy', i)

    print('Done.')


