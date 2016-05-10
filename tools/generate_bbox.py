import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import os
import dlib
import scipy.io as sio
from skimage import io
import numpy as np

def run_dlib_selective_search(image_name):
    img = io.imread(image_name)
    rects = []
    dlib.find_candidate_object_locations(img,rects,min_size=0)
    proposals = []
    for k,d in enumerate(rects):
        #templist = [d.left(),d.top(),d.right(),d.bottom()]
        templist = [d.top(),d.left(),d.bottom(),d.right()]
        proposals.append(templist)
    proposals = np.array(proposals)
    return proposals

imagenet_path = '/path/to/imagenet/data/Images/'
names = 'path/to/imagenet/data/ImageSets/train.txt'

count = 0
all_proposals = []
imagenms = []
nameFile = open(names)
for line in nameFile.readlines():
	filename = imagenet_path+line.split('\n')[0]+'.png'
	single_proposal = run_dlib_selective_search(filename)
	all_proposals.append(single_proposal)
	count = count+1;
	print count

sio.savemat('train.mat',mdict={'all_boxes':all_proposals,'images':imagenms})
obj_proposals = sio.loadmat('train.mat')
print obj_proposals

