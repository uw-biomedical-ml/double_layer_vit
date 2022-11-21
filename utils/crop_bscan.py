import cv2
import numpy as np
import os
import glob
import random
import argparse

def crop_bscan(im, output_size):

    h,w = im.shape[0], im.shape[1]
    middle = np.argmax(np.sum(im[:,:,0], axis=1))
    if middle - int(output_size[0]/2) < 0:
        im_crop = im[:output_size[0]]
    elif middle + int(output_size[0]/2) > h:
        im_crop = im[h-output_size[0]:]
    else:
        im_crop = im[middle-int(output_size[0]/2):middle+int(output_size[0]/2)]
    assert(im_crop.shape[0]==output_size[0])
    return im_crop


if __name__ == '__main__':
    output_size = (768, 500)

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', required=True, type=str)
    parser.add_argument('-o', '--outputdir', required=True, type=str)

    args = parser.parse_args()
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)

    print(os.path.join(args.inputdir, "**/*.png"))
    fpaths = sorted(glob.glob(os.path.join(args.inputdir, "**/*.png"), recursive=True))
    #fpaths = sorted(glob.glob(os.path.join(args.inputdir, "*.png")))
    print(len(fpaths))

    for fpath in fpaths:
        im = cv2.imread(fpath)
        im_crop = crop_bscan(im, output_size)
        fname = fpath.split("/")[-1]
        cv2.imwrite(os.path.join(args.outputdir, fname.replace('png', 'jpg')), im_crop)
