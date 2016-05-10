#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')

import _init_paths
from fast_rcnn.config import cfg
from fast_rcnn.test import im_detect
from utils.cython_nms import nms
from utils.timer import Timer
import matplotlib.pyplot as plt
import numpy as np
import scipy.io as sio
import caffe, os, sys, cv2
import argparse
import dlib
from skimage import io
'''
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
CLASSES = ('__background__','person')

NETS = {'vgg16': ('VGG16',
                  'vgg16_fast_rcnn_iter_40000.caffemodel'),
        'vgg_cnn_m_1024': ('VGG_CNN_M_1024',
                           'vgg_cnn_m_1024_fast_rcnn_iter_40000.caffemodel'),
        'caffenet': ('CaffeNet',
                     'caffenet_fast_rcnn_iter_40000.caffemodel')}


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return

    im = im[:, :, (2, 1, 0)]
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(im, aspect='equal')
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
            )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                  fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

def run_dlib_selective_search(image_name):
    img = io.imread(image_name)
    rects = []
    dlib.find_candidate_object_locations(img,rects,min_size=50)
    proposals = []
    for k,d in enumerate(rects):
        templist = [d.left(),d.top(),d.right(),d.bottom()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

def demo(net, im_file, classes):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load pre-computed Selected Search object proposals
    #box_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo',image_name + '_boxes.mat')
    #obj_proposals = sio.loadmat(box_file)['boxes']
    #im_file = os.path.join(cfg.ROOT_DIR, 'data', 'demo', image_name)
    obj_proposals = run_dlib_selective_search(im_file)

    # Load the demo image
    im = cv2.imread(im_file)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(net, im, obj_proposals)
    timer.toc()
    print ('Detection took {:.3f}s for '
           '{:d} object proposals').format(timer.total_time, boxes.shape[0])
    
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls in classes:
		cls_ind = CLASSES.index(cls)
		cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]

		cls_scores = scores[:, cls_ind]
		keep = np.where(cls_scores >= CONF_THRESH)[0]
		try:
			cls_boxes = cls_boxes[keep, :]
			cls_scores = cls_scores[keep]
			dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
			keep = nms(dets, NMS_THRESH)
			dets = dets[keep, :]
		except Exception,e:
			dets = np.hstack((cls_boxes,cls_scores[:, np.newaxis])).astype(np.float32)
		print 'All {} detections with p({} | box) >= {:.1f}'.format(cls, cls,
		                                                       CONF_THRESH)
		if 0:
		    vis_detections(im, cls, dets, thresh=CONF_THRESH)
		return im,dets

def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Train a Fast R-CNN network')
    parser.add_argument('--gpu', dest='gpu_id', help='GPU device id to use [0]',
                        default=0, type=int)
    parser.add_argument('--cpu', dest='cpu_mode',
                        help='Use CPU mode (overrides --gpu)',
                        action='store_true')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16]',
                        choices=NETS.keys(), default='caffenet')

    args = parser.parse_args()

    return args

def load_inria_annotation(index):
    """
    Load image and bounding boxes info from txt files of INRIAPerson.
    """
    filename = os.path.join('/home/xuetingli/test/INRIA/data/Annotations', index.split('.')[0] + '.txt')
    # print 'Loading: {}'.format(filename)
    with open(filename) as f:
            data = f.read()
    import re
    objs = re.findall('\(\d+, \d+\)[\s\-]+\(\d+, \d+\)', data)
    
    num_objs = len(objs)

    boxes = np.zeros((num_objs, 4), dtype=np.uint16)

    # Load object bounding boxes into a data frame.
    for ix, obj in enumerate(objs):
        # Make pixel indexes 0-based
        coor = re.findall('\d+', obj)
        x1 = float(coor[0])
        y1 = float(coor[1])
        x2 = float(coor[2])
        y2 = float(coor[3])
        boxes[ix, :] = [x1, y1, x2, y2]


    return boxes
        
if __name__ == '__main__':
	args = parse_args()

	prototxt = os.path.join(cfg.ROOT_DIR, 'models', NETS[args.demo_net][0],
	                        'test.prototxt')
	#print prototxt
	caffemodel = os.path.join(cfg.ROOT_DIR, 'data', 'fast_rcnn_models',
	                          NETS[args.demo_net][1])

	if not os.path.isfile(caffemodel):
	    raise IOError(('{:s} not found.\nDid you run ./data/script/'
	                   'fetch_fast_rcnn_models.sh?').format(caffemodel))

	if args.cpu_mode:
	    caffe.set_mode_cpu()
	else:
	    caffe.set_mode_gpu()
	    caffe.set_device(args.gpu_id)
	net = caffe.Net(prototxt, caffemodel, caffe.TEST)

	print '\n\nLoaded network {:s}'.format(caffemodel)
	#demo(net,'/home/xuetingli/test/INRIA/data/Images/crop001001.png',('person',))
	#demo(net,'/home/xuetingli/test/INRIA/data/Images/crop001002.png',('person',))
	#demo(net,'/home/xuetingli/test/INRIA/data/Images/crop001003.png',('person',))

	imgPath = '/home/xuetingli/test/INRIA/data/Images/'
	fp = 0
	tp = 0
	totalGTBoxes = 0

	for parent,dirnames,filenames in os.walk(imgPath):
		for imgnm in filenames:

			#imgnm = 'crop001008'
			[im,dets] = demo(net,imgPath+imgnm,('person',))

			inds = np.where(dets[:, -1] >= 0.8)[0]

			if not (len(inds)==0):        
				#load ground truth boxes
				boxes = load_inria_annotation(imgnm)
				BeenChoosen = [1 for i in range(len(dets))]
				totalGTBoxes = totalGTBoxes + len(boxes)
				imax = 0
				for bbgt in boxes:
					#loop over dets to find maximum overlap
					ovmax = -1000000
					for i in range(0,len(dets)):
						det = dets[i]
						bb = det[0:4]
						score = det[-1]
						bi = [max(bb[0],bbgt[0]),max(bb[1],bbgt[1]),min(bb[2],bbgt[2]),min(bb[3],bbgt[3])]
						iw = bi[2]-bi[0]+1
						ih = bi[3]-bi[1]+1
						if iw>0 and ih >0:
							ua=(bb[2]-bb[0]+1)*(bb[3]-bb[1]+1)+(bbgt[2]-bbgt[0]+1)*(bbgt[3]-bbgt[1]+1)-iw*ih;
							ov = iw*ih/ua;
							if ov > ovmax:
								ovmax = ov
								imax = i
					print 'ovmax',ovmax
					if ovmax > 0.5:
						tp = tp + 1
						BeenChoosen[imax] = 0
				fp = fp + sum(BeenChoosen)
	print 'the number of true positive is',tp
	print 'the number of false positive is',fp
	print 'the number of total bboxes is ',totalGTBoxes
        
	plt.show()
