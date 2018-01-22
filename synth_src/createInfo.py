import multiprocessing
from skimage.util import img_as_float
from skimage import io
from skimage.segmentation import slic
import numpy as np
import time
import cv2

cam_data = np.load('./cam_data.npy')
# camName,camPara,p3d,visTable
cam_name = cam_data[0]
cam_para = cam_data[1]
p3d = cam_data[2]
visTable = cam_data[3]


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


if __name__ == '__main__':

    USE_POOL = False

    if USE_POOL:
        pool = multiprocessing.Pool(processes=8)
        for i in range(0,8):
            pool.apply_async(saveSlic, (i,))
        pool.close()
        pool.join()


    print('Done.')







