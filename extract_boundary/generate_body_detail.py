#coding=utf-8
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

def split_map(datapath, datapath1):
    print(datapath)
    for name in os.listdir(datapath):
        mask = cv2.imread(datapath+'/'+name,0)
        # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # mask = np.int64(mask > 32)
        body = cv2.blur(mask, ksize=(5,5))
        body = cv2.distanceTransform(body, distanceType=cv2.DIST_L2, maskSize=5)
        body = body**0.5

        tmp  = body[np.where(body>0)]
        if len(tmp)!=0:
            body[np.where(body>0)] = np.floor(tmp/np.max(tmp)*255)

        # datapath1 = '/home/david/dataset/ProstateSeg/Prostate_move_black/'

        if not os.path.exists(datapath1+'/body/'):
            os.makedirs(datapath1+'/body/')
        cv2.imwrite(datapath1+'/body/'+name, body)

        if not os.path.exists(datapath1+'/detail/'):
            os.makedirs(datapath1+'/detail')
        cv2.imwrite(datapath1+'/detail/'+name, mask-body)


if __name__=='__main__':
    datapath = '/home/david/datasets/ProstateSeg/Prostate'
    split_map(os.path.join(datapath,"masks/train"), datapath)
