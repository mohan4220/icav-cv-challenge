
import cv2
import pickle
import pandas as pd
import numpy as np
import os
import xml.etree.ElementTree as et
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard


def get_iou(bb1, bb2):
    """ function that return intersetion over union percentage"""
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


# paths to xml annotations, images and converted csv annotations
path = "dataset/pascal/"
imgpath = "dataset/images/"
annot = "dataset/csv/"

filenames = [f.split('.')[0] for f in os.listdir(path)]
# image = cv2.imread(os.path.join(imgpath,filenames[0]+'.jpg'))
cv2. setUseOptimized(True)
cv2.setNumThreads(4)
# creating a selective search segmentation object
ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
train_images = []
train_labels = []
# creating dataset and lables to train, converted xml annotations to csv since csv files are faster to read compared to xml files
for e, i in enumerate(os.listdir(annot)):
    try:
        filename = i.split(".")[0]+".jpg"
        print(e, filename, "===>>", "{:.1f}%".format((e+1)*100/139))
        image = cv2.imread(os.path.join(imgpath, filename))
        df = pd.read_csv(os.path.join(annot, i))
        gtvalues = []
        for row in df.iterrows():
            x1 = row[1][0]
            y1 = row[1][1]
            x2 = row[1][2]
            y2 = row[1][3]
            gtvalues.append({"x1": x1, "x2": x2, "y1": y1, "y2": y2})

        ss.setBaseImage(image)
        ss.switchToSelectiveSearchFast()
        ssresults = ss.process()
        imout = image.copy()
        counter = 0
        falsecounter = 0
        flag = 0
        fflag = 0
        bflag = 0
        # ===================================
        print("trying to reduce rectangles")
        new = []
        for n, j in enumerate(ssresults):
            new.append((n, j[2]*j[3]))

        new.sort(key=lambda x: x[1])
        new.reverse()
        filename = filename.split('.')[0]
        csv = pd.read_csv(annot+filename+'.csv')
        min_area = min([(row[1][2]-row[1][0])*(row[1][3]-row[1][1])
                        for row in csv.iterrows()])
        print('minimum area:', min_area)
        new = [j for j in new if j[1] >= min_area]
        results = []
        for n, j in new:
            # print(ssresults[n])
            results.append(ssresults[n])
        print('done reducing rectangles')
        # ===================================
        for e, result in enumerate(ssresults):
            if e < 2000 and flag == 0:
                for gtval in gtvalues:
                    x, y, w, h = result
                    iou = get_iou(
                        gtval, {"x1": x, "x2": x+w, "y1": y, "y2": y+h})
                    if counter < 30:
                        if iou > 0.70:
                            timage = imout[y:y+h, x:x+w]
                            resized = cv2.resize(
                                timage, (128, 128), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(1)
                            counter += 1
                            # cv2.namedWindow("new")
                            # cv2.imshow("new", resized)
                            # cv2.waitKey(200)
                    else:
                        fflag = 1
                    if falsecounter < 30:
                        if iou < 0.3:
                            timage = imout[y:y+h, x:x+w]
                            resized = cv2.resize(
                                timage, (128, 128), interpolation=cv2.INTER_AREA)
                            train_images.append(resized)
                            train_labels.append(0)
                            falsecounter += 1
                    else:
                        bflag = 1
                if fflag == 1 and bflag == 1:
                    print("inside")
                    flag = 1
    except Exception as e:
        print(e)
        print("error in "+filename)
        continue

X_new = np.array(train_images)
y_new = np.array(train_labels)

# pickling the resulting dataset foe later use
infile = open('x.pickle', 'wb')
pickle.dump(X_new, infile)
infile.close()
print('x dumped')

infile = open('y.pickle', 'wb')
pickle.dump(y_new, infile)
infile.close()
print('y dumped')
# obj = pickle.load(infile)
sfile = bz2.BZ2File('xsml.pickle', 'w')
pickle.dump(X_new, sfile)
sfile.close()
print('xsml dumped')
