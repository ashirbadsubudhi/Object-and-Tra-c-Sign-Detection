import numpy as np
from keras.models import load_model
import keras.backend as K

obj_thres = 0.5
box_thres = 0.4
model = load_model('data/yolo.h5')

def sigmoid(x):
	return 1 / (1 + np.exp(-x))

def processing(out, anchors, mask):
	grid_h, grid_w, num_boxes = map(int, out.shape[1: 4])

	anchors = [anchors[i] for i in mask]
	anchors_tensor = np.array(anchors).reshape(1, 1, len(anchors), 2)

	out = out[0]
	box_xy = sigmoid(out[..., :2])
	box_wh = np.exp(out[..., 2:4])
	box_wh = box_wh * anchors_tensor

	box_confidence = sigmoid(out[..., 4])
	box_confidence = np.expand_dims(box_confidence, axis=-1)
	box_class_probs = sigmoid(out[..., 5:])

	col = np.tile(np.arange(0, grid_w), grid_w).reshape(-1, grid_w)
	row = np.tile(np.arange(0, grid_h).reshape(-1, 1), grid_h)

	col = col.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	row = row.reshape(grid_h, grid_w, 1, 1).repeat(3, axis=-2)
	grid = np.concatenate((col, row), axis=-1)

	box_xy += grid
	box_xy /= (grid_w, grid_h)
	box_wh /= (416, 416)
	box_xy -= (box_wh / 2.)
	boxes = np.concatenate((box_xy, box_wh), axis=-1)

	return boxes, box_confidence, box_class_probs

def boxes_filter(boxes, box_confidences, box_class_probs):
	box_scores = box_confidences * box_class_probs
	box_classes = np.argmax(box_scores, axis=-1)
	box_class_scores = np.max(box_scores, axis=-1)
	pos = np.where(box_class_scores >= obj_thres)

	boxes = boxes[pos]
	classes = box_classes[pos]
	scores = box_class_scores[pos]

	return boxes, classes, scores

def boxes_prediction(boxes, scores):
	x = boxes[:, 0]
	y = boxes[:, 1]
	w = boxes[:, 2]
	h = boxes[:, 3]

	areas = w * h
	order = scores.argsort()[::-1]

	keep = []
	while order.size > 0:
		i = order[0]
		keep.append(i)

		xx1 = np.maximum(x[i], x[order[1:]])
		yy1 = np.maximum(y[i], y[order[1:]])
		xx2 = np.minimum(x[i] + w[i], x[order[1:]] + w[order[1:]])
		yy2 = np.minimum(y[i] + h[i], y[order[1:]] + h[order[1:]])

		w1 = np.maximum(0.0, xx2 - xx1 + 1)
		h1 = np.maximum(0.0, yy2 - yy1 + 1)
		inter = w1 * h1

		ovr = inter / (areas[i] + areas[order[1:]] - inter)
		inds = np.where(ovr <= box_thres)[0]
		order = order[inds + 1]

	keep = np.array(keep)

	return keep

def yolo_out(outs, shape):
	masks = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
	anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45],
                   [59, 119], [116, 90], [156, 198], [373, 326]]

	boxes, classes, scores = [], [], []

	for out, mask in zip(outs, masks):
		b, c, s = processing(out, anchors, mask)
		b, c, s = boxes_filter(b, c, s)
		boxes.append(b)
		classes.append(c)
		scores.append(s)

	boxes = np.concatenate(boxes)
	classes = np.concatenate(classes)
	scores = np.concatenate(scores)

	width, height = shape[1], shape[0]
	im_dims = [width, height, width, height]
	boxes = boxes * im_dims

	nboxes, nclasses, nscores = [], [], []
	for c in set(classes):
		inds = np.where(classes == c)
		b = boxes[inds]
		c = classes[inds]
		s = scores[inds]

		keep = boxes_prediction(b, s)

		nboxes.append(b[keep])
		nclasses.append(c[keep])
		nscores.append(s[keep])

	if not nclasses and not nscores:
		return None, None, None

	boxes = np.concatenate(nboxes)
	classes = np.concatenate(nclasses)
	scores = np.concatenate(nscores)

	return boxes, classes, scores

def prediction(image, shape):
	output = model.predict(image)
	boxes, classes, scores = yolo_out(output, shape)
	return boxes, classes, scores