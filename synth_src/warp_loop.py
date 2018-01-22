
import numpy as np
import cv2


class Image:

    def __init__(self,name):
        if isinstance(name,str):
            self.name = name
            self.img = cv2.imread(name)
        else:
            print('ERROR, name should bne string.')

    def show(self):
        rows,cols,channel = self.img.shape
        if rows > 1000 and cols > 1000:
            imgin = cv2.resize(self.img,(cols//2,rows//2),interpolation=cv2.INTER_CUBIC)
        else:
            imgin = self.img
        cv2.imshow("img",imgin)
        cv2.waitKey(0)

    def setImage(self,img):
        self.img = img

    def setCameraPos(self,pos):
        self.Q = pos[0] # Q[0]:f,Q[1]:x_c, Q[2]:y_c
        self.R = pos[1] # 3x3
        self.T = pos[2] # 3x1

    def setTargetPos(self,pos):
        self.targetR = pos[0]
        self.targetT = pos[1]

    def LoopCameraForWarp(self):
        pass

if __name__ == "__main__":
    use_real_cam_number = 3
    cam_data = np.load('./cam_data.npy')
    # camName,camPara,p3d,visTable
    cam_name = cam_data[0]
    cam_para = cam_data[1]

    targetPos = [np.array([[ 0.93045067,  0.01840559, -0.36595473],
                           [ 0.07159506,  0.97035553,  0.23083646],
                           [ 0.35935479, -0.2409824 ,  0.9015499 ]]),
                 np.array([[ 1.2568937 ],
                           [ 0.01163087],
                           [ 0.14351989]])]
    ind = np.array([6,7,5])
    for j in range(0,use_real_cam_number):
        imgname = cam_name[ind[j]][0]
        imgpos = cam_para[ind[j]] # [] 0:Q(f,x,y) 1:R 2:T
        img = Image(imgname)
        img.setCameraPos(imgpos)
        img.setTargetPos(targetPos)




