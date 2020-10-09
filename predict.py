import cv2
import tensorflow as tf
import sys
import numpy as np


def non_max_suppression_fast(boxes, overlapThresh):
    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []
    # if the bounding boxes integers, convert them to floats --
    # this is important since we'll be doing a bunch of divisions
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    # initialize the list of picked indexes
    pick = []
    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)
    # keep looping while some indexes still remain in the indexes
    # list
    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last],
                                               np.where(overlap > overlapThresh)[0])))
    # return only the bounding boxes that were picked using the
    # integer data type
    return boxes[pick].astype("int")


# =======================================================================================
img = cv2.imread(sys.argv[1])
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)
ss.switchToSelectiveSearchFast()

rects = ss.process()
rects = non_max_suppression_fast(rects, 0.3)

imgOut = img.copy()
model = tf.keras.models.load_model("kangaroo_cnn.model")
# print(rects.shape)
predictions = []
out_rects = []
out_predictions = []
count = 0
for i, rect in enumerate(rects):
    # print(rect)
    x, y, w, h = rect
    # cv2.rectangle(imgOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    image = cv2.resize(imgOut[y:y+h, x:x+w], (128, 128),
                       interpolation=cv2.INTER_AREA)
    image = image.reshape(-1, 128, 128, 3)
    # cv2.imshow("new", image)
    # cv2.waitKey(500)
    prediction = model.predict([image])
    predictions.append(prediction)
    if predictions[i][0][0] > 0.80:
        count = count+1
        out_predictions.append(float(predictions[i][0][0]))
        out_rects.append(rect)
        cv2.rectangle(imgOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
    # print(i, predictions[i])
    # print(i, image.shape)

out_rects = non_max_suppression_fast(np.array(out_rects), 0.7)
# print(rects.shape, type(rects), out_rects.shape)

cv2.imshow("new", imgOut)
cv2.waitKey(0)
print("count: ", count)
