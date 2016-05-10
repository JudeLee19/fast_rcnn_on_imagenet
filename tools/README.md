## Running demo.py using dlib's selective search:

I changed some code in demo.py to use dlib’s selective search to generate proposal bounding boxes, so no *_boxes.mat is needed for the demo or other pictures.

To run the changed demo, you need to compile dlib for python:

1. Download [dlib](http://dlib.net/)
2. if you are on ubuntu:

        sudo apt-get install libboost-python-dev cmake
        cd dlib-18.15/python_examples
        ./compile_dlib_python_module.bat 
        sudo cp dlib.so /usr/local/lib/python2.7/dist-packages/
   If import dlib in python shell gives no error, the dlib shoud be ready.
   
3. Then just install fast-rcnn and run demo.py. A dog should be detected in the shown image. Change the image to your own to run detection by fast-rcnn(only the 20 classes in Pascal Voc).

## Training and testing on Imagenet without MATLAB
Please reference to my [blog](http://sunshineatnoon.github.io/Train-fast-rcnn-model-on-imagenet-without-matlab/).
