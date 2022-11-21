import cv2
import os
import numpy as np
import glob
import argparse


def enface_proj(predmaks, eye_id, output_size):

    canvas = np.zeros((output_size[0], output_size[1], 3)) # SS_COT
    for predmask in predmasks:
        if len(predmask.split("/")[-1].split("_")) != 8:
            continue
        pid_eye = predmask.split("/")[-1].split("_")[0]+"_"+predmask.split("/")[-1].split("_")[4]
        if eye_id != pid_eye:
            continue
        s = int(predmask.split("/")[-1].split('.')[0].split("_")[-1][-3:]) # B-scan slice id
        im = cv2.imread(predmask)
        assert im.shape[1]==output_size[0]

        # decode colors
        decoded_mask = np.zeros((im.shape[0], im.shape[1], 2))
        c = np.argmin(im, axis=-1)
        decoded_mask[:,:,0] = c==0 # Drusen
        decoded_mask[:,:,1] = c==2 # DLS
        # project value
        proj_mask = np.zeros((im.shape[1], 2))
        proj_mask[:,0] = np.sum(decoded_mask[:,:,0], axis=0) # Drusen
        proj_mask[:,1] = np.sum(decoded_mask[:,:,1], axis=0) # DLS

        canvas_s = np.argmax(proj_mask, axis=-1)
        canvas_s_zero = (np.sum(proj_mask, axis=-1)==0)
        canvas_s_color = np.zeros((canvas_s.shape[0], 3))
        canvas_s_color[canvas_s == 0] = [255,0,0] # Drusen
        canvas_s_color[canvas_s == 1] = [0,255,0] # DLS
        canvas_s_color[canvas_s_zero == 1] = [0,0,255] #bg

        canvas[s] = canvas_s_color

    return canvas




if __name__ == '__main__':
    output_size = (500, 500) # for SS_OCT

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--inputdir', required=True, type=str, help='directory where the prediction masks (must be the same name as original Bscans) are stored.') # directory where the predictions are stored
    parser.add_argument('-o', '--outputdir', required=True, type=str, help='directory path to store enface map.') # directory where the map will be generated
    parser.add_argument('-e', '--eyeid', required=True, type=str, help='specify eyeid(ex. "P1251_OD") to process.') # eye_id ex. "P1251_OD"

    args = parser.parse_args()
    if not os.path.isdir(args.outputdir):
        os.makedirs(args.outputdir)

    predmasks = sorted(glob.glob(os.path.join(args.inputdir, "*.jpg")))

    map = enface_proj(predmasks, args.eyeid, output_size)
    cv2.imwrite("%s/%s.png"%(args.outputdir, args.eyeid), map)
