import numpy as np
import torch

# ---------------------------- NMS ----------------------------
## basic NMS
def nms(bboxes, scores, nms_thresh):
    """"Pure Python NMS."""
    x1 = bboxes[:, 0]  #xmin
    y1 = bboxes[:, 1]  #ymin
    x2 = bboxes[:, 2]  #xmax
    y2 = bboxes[:, 3]  #ymax

    areas = (x2 - x1) * (y2 - y1)
    order = torch.argsort(scores, descending=True)

    keep = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        # compute iou
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])

        w = torch.max(torch.ones_like(xx1) * 1e-10, xx2 - xx1)
        h = torch.max(torch.ones_like(xx1) * 1e-10, yy2 - yy1)
        inter = w * h

        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
        #reserve all the boundingbox whose ovr less than thresh
        inds = torch.where(iou <= nms_thresh)[0]
        order = order[inds + 1]

    return keep

## class-aware NMS 
def multiclass_nms(scores, labels, bboxes, nms_thresh, num_classes):
    # nms
    keep_by_nms = torch.zeros(len(bboxes)).long()
    for i in range(num_classes):
        inds = torch.where(labels == i)[0]
        if len(inds) == 0:
            continue
        c_bboxes = bboxes[inds]
        c_scores = scores[inds]
        c_keep = nms(c_bboxes, c_scores, nms_thresh)
        keep_by_nms[torch.index_select(inds, 0, torch.stack(c_keep))] = 1
        # keep_by_nms[inds[c_keep]] = 1
    keep_by_nms = torch.where(keep_by_nms > 0)[0]

    return keep_by_nms
