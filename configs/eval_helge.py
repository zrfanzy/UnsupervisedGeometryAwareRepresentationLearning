import scipy.io
import numpy as np
import cv2
import glob
import os
import glob
from data_generation import *
import param
from param import *
from skimage.measure import compare_psnr, compare_ssim


countnum = np.zeros((15,15))
psnrs = np.zeros((15,15))
ssims = np.zeros((15,15))

scale_factor = 1.14/2
img_width = 128
img_height = 128

imgfiles = glob.glob('/scratch/rzhou/posewarp-cvpr2018/poses/All/*.jpg')
imgfiles = sorted(imgfiles)

count = 69 * 15
target = []
for i in range(15):
    img1 = imgfiles[count]
    base = os.path.basename(img1)
    filename = os.path.splitext(base)[0]
    matfile1 = '/scratch/rzhou/posewarp-cvpr2018/poses/pose/' + filename + '.mat'
    x = sio.loadmat(matfile1)
    target.append([x['box'], x['X'], img1])
    count = count + 1

count = 0
for i in range(69):
    for j in range(15):
        img = imgfiles[count]
        base = os.path.basename(img)
        filename = os.path.splitext(base)[0]
        print(str(count) + " : filename: " + filename)
        matfile = '/scratch/rzhou/posewarp-cvpr2018/poses/pose/' + filename + '.mat'

        if (not os.path.isfile(matfile)):
            count = count + 1
            continue

        x1 = scipy.io.loadmat(matfile)
        target.append([x['box'], x['X'], img1])
        count = count + 1

        for k in range(15):
            # person i, pose j -> pose k

            # get scale
            I0, joints0, scale0, pos0 = read_frame_ours(img, x1['box'][0], x1['X'])
            I1, joints1, scale1, pos1 = read_frame_ours(target[i][2], target[i][0][0], target[i][1])

            if scale0 > scale1:
                scale = scale_factor / scale0
            else:
                scale = scale_factor / scale1

            pos = (pos0 + pos1) / 2.0


            # get groundtruth
            count1 = (i) * 15 + k
            img1 = imgfiles[count1]
            base1 = os.path.basename(img1)
            filename1 = os.path.splitext(base1)[0]
            matfile1 = '/scratch/rzhou/posewarp-cvpr2018/poses/pose/' + filename1 + '.mat'
            
            if (not os.path.isfile(matfile1)):
                continue
            x = scipy.io.loadmat(matfile1)
            targett=([x['box'], x['X'], img1])
            I1, joints1, scale1, pos1 = read_frame_ours(targett[2], targett[0][0], targett[1])
            I1, joints1 = center_and_scale_image(I1, img_width, img_height, pos, scale, joints1)
            cv2.imwrite('groundtruth/' + filename + '_' + str(k) + '.png', I1)
            #cv2.imwrite('gd.png', I1)
            #cv2.imwrite('gd.png', I1)
            
            # get network output
            #tmp = cv2.imread('results/' + filename + '_' + str(k) + '.png')
            #cv2.imwrite('output.png', tmp)
            #tmp = np.reshape(tmp, (256,256,3))
            #tmp = (tmp * 255).astype('int')
            #cv2.imwrite('output.png', tmp)

            #psnr = compare_psnr(I1, tmp)
            #ssim = compare_ssim(I1, tmp, multichannel=True)
            #psnrs[j][k] = psnrs[j][k] + psnr
            #ssims[j][k] = ssims[j][k] + ssim
            #print((psnr,ssim))
            #break
            #countnum[j][k] = countnum[j][k] + 1
            #np.save('compare.npy', [psnrs, ssims, countnum])
#np.save('compare.npy', [psnrs, ssims, countnum])
        