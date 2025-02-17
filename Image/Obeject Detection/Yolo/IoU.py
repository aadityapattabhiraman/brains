#!/home/akugyo/Programs/Python/torch/bin/python

import torch


def intersection_over_boxes(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates Intersection Over Union.

    Parameters:
        boxes_preds (tensor): Predictions of bounding boxes (batch_size, 4)
        boxes_labels (tensor): Correct labels of bounding boxes (batch_size, 4)
        box_format (str): midpoint/corners, if boxes (x, y, w, h) or
            (x1, y1, x2, y2)

    Returns:
        tensor: Intersection over union for all examples
    """
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - box_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - box_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + box_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + box_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - box_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - box_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + box_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + box_labels[..., 3:4] / 2

    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torxh.max(box1_x1, box2_x1)
    y1 = torxh.max(box1_y1, box2_y1)
    x2 = torxh.max(box1_x2, box2_x2)
    y2 = torxh.max(box1_y1, box2_y1)

    intersection = (x2 - x1).clmap(0) * (y2 -y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y1 - box1_y2))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y1 - box2_y2))

    return intersection / (box1_area + box2_area - intersection + 1e-6)
