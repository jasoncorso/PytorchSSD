# jason playing with the PytorchSSD code
# get it to load a model and process an image?
#
# This is basically a single image and fixed model version of the live.py demo

import sys
from collections import OrderedDict

import numpy as np
import torch
from torch.autograd import Variable

sys.path.append('../')
sys.path.append('../models/')
sys.path.append('../layers/')
sys.path.append('../utils/')
sys.path.append('../data/')

import models.SSD_vgg as SSD
from data import VOC_300
from layers.functions import Detect, PriorBox
from utils.timer import Timer

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


class ObjectDetector:
    def __init__(self, net, detection, transform, num_classes=21, cuda=False, max_per_image=300, thresh=0.5):
        self.net = net
        self.detection = detection
        self.transform = transform
        self.max_per_image = 300
        self.num_classes = num_classes
        self.max_per_image = max_per_image
        self.cuda = cuda
        self.thresh = thresh

    def predict(self, img):
        scale = torch.Tensor([img.shape[1], img.shape[0],
                              img.shape[1], img.shape[0]]).cpu().numpy()
        _t = {'im_detect': Timer(), 'misc': Timer()}
        assert img.shape[2] == 3
        x = Variable(self.transform(img).unsqueeze(0), volatile=True)
        if self.cuda:
            x = x.cuda()
        _t['im_detect'].tic()
        out = net(x, test=True)  # forward pass
        boxes, scores = self.detection.forward(out, priors)
        detect_time = _t['im_detect'].toc()
        boxes = boxes[0]
        scores = scores[0]

        boxes = boxes.cpu().numpy()
        scores = scores.cpu().numpy()
        # scale each detection back up to the image
        boxes *= scale
        _t['misc'].tic()
        all_boxes = [[] for _ in range(num_classes)]

        for j in range(1, num_classes):
            inds = np.where(scores[:, j] > self.thresh)[0]
            if len(inds) == 0:
                all_boxes[j] = np.zeros([0, 5], dtype=np.float32)
                continue
            c_bboxes = boxes[inds]
            c_scores = scores[inds, j]
            print(scores[:, j])
            c_dets = np.hstack((c_bboxes, c_scores[:, np.newaxis])).astype(
                np.float32, copy=False)
            # keep = nms(c_bboxes,c_scores)

            keep = py_cpu_nms(c_dets, 0.45)
            keep = keep[:50]
            c_dets = c_dets[keep, :]
            all_boxes[j] = c_dets
        if self.max_per_image > 0:
            image_scores = np.hstack([all_boxes[j][:, -1] for j in range(1, num_classes)])
            if len(image_scores) > self.max_per_image:
                image_thresh = np.sort(image_scores)[-self.max_per_image]
                for j in range(1, num_classes):
                    keep = np.where(all_boxes[j][:, -1] >= image_thresh)[0]
                    all_boxes[j] = all_boxes[j][keep, :]

        nms_time = _t['misc'].toc()
        print('net time: ', detect_time)
        print('post time: ', nms_time)
        return all_boxes



img_dim = 300
num_classes = 21  # the model says it is 7+12, but this is not the class count,
                  #  it is the years of the VOC datasets.  total classes are 21

priorbox = PriorBox(VOC_300)
# noting the interpreter message that volatile=True no longer has any effect
# and it should rather be done with `with torch.no_grad()`
priors = Variable(priorbox.forward(), volatile=True)

net = SSD.build_net(img_dim, num_classes)

# the hackiness in the stripping the `module` here is needed and from the
# example, but I do not understand its source by any means.  There's some
# trickery afoot.
state_dict = torch.load(
    "/scratch/jason-model-cache/SSD_vgg_VOC_epoches_270.pth")
#net.load_state_dict(state_dict) does not work for this model because there are
# extra .module pieces in the pth file-dict
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    head = k[:7]
    if head == 'module.':
        name = k[7:]  # remove `module.`
    else:
        name = k
    print("operating on %s --> %s" % (k, name))
    new_state_dict[name] = v
net.load_state_dict(new_state_dict)
net.eval()

print("finished loading the model")
print(net)




