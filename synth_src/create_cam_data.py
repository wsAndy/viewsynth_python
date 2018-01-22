
import numpy as np

path = 'G:\\viewsynth_with_superpixel_fusion\\PMVS\\data\\museum.nvm.cmvs\\00\\';


## Parse_cam

fp = open(path+'cameras_v2.txt','r')
line = fp.readline()
while line:
    if line == '# The nubmer of cameras in this reconstruction\n':
        break;
    line = fp.readline()

N = fp.readline()
N = int(N[0:-1])

camPara = []
camName = []

for i in range(0,N,1):
    curName = str(i).zfill(8)+'.jpg'
    line = fp.readline()
    while line[0:-1] != curName:
        line = fp.readline()
    line = fp.readline() # path
    q1 = (fp.readline())[0:-1] # Q1
    q1 = float(q1)
    [q2,q3] = (fp.readline())[0:-1].split(' ') # q2  q3
    q2 = float(q2)
    q3 = float(q3)
    [t1,t2,t3] = (fp.readline())[0:-1].split(' ') # t1 t2 t3
    t1 = float(t1)
    t2 = float(t2)
    t3 = float(t3)
    # jump useless line
    fp.readline()
    fp.readline()
    fp.readline()
    r1_line = (fp.readline())[0:-1].split(' ')
    r2_line = (fp.readline())[0:-1].split(' ')
    r3_line = (fp.readline())[0:-1].split(' ')
    r1_line = [float(r1_line[0]), float(r1_line[1]), float(r1_line[2])]
    r2_line = [float(r2_line[0]), float(r2_line[1]), float(r2_line[2])]
    r3_line = [float(r3_line[0]), float(r3_line[1]), float(r3_line[2])]
    Q = np.array([ [q1],[q2],[q3] ])
    R = np.array([r1_line, r2_line, r3_line])
    T = np.array([ [t1],[t2],[t3] ])
    camPara.append([Q,R,T])
    camName.append(path+'visualize\\'+curName)

fp.close()
## for patch

fp = open(path+'models\\option-0000.patch','r');
line = fp.readline()
M = fp.readline()
M = int(M[0:-1])
p3d = []
visTable = set([str(x) for x in range(0, N, 1)])

for index in range(0,M,1):
    line = fp.readline()
    while line[0:-1] != 'PATCHS':
        line = fp.readline()
    [p1,p2,p3,p4] = (fp.readline())[0:-1].split(' ')
    p1 = float(p1)
    p2 = float(p2)
    p3 = float(p3)
    p4 = float(p4)
    p3d_ = np.array([ [p1], [p2], [p3] ])
    fp.readline()
    fp.readline()
    fp.readline()
    vis = (fp.readline())[0:-1].split(' ')
    vis = set(vis)
    p3d.append(p3d_)
    visTable = visTable - vis

visTable = set([str(x) for x in range(0, N, 1)]) - visTable
visTable = sorted([int(x) for x in visTable])

camPara_ = []
camName_ = []
p3d_ = []
for x in visTable:
    camName_.append([camName[x]])
    camPara_.append(camPara[x])
    p3d_.append(p3d[x])
#

# print(p3d_.__len__())
# print(visTable.__len__())
# print(camPara_.__len__())
# print(camName_.__len__())

cam_data = np.array([camName_, camPara_, p3d_, visTable])

np.save('./cam_data.npy',cam_data)
