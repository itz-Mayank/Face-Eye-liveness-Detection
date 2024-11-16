# Face-Eye-liveness-Detection
# Installation

## Upgrade pip (optional but recommended):
pip install --upgrade pip

## Install OpenCV:
pip install opencv-python
pip install opencv-contrib-python

## Install TensorFlow:
pip install tensorflow

## Install NumPy (used for numerical operations in TensorFlow and OpenCV):
pip install numpy



## Create a virtual environment
python -m venv venv
venv\Scripts\activate

## If you are using GPU
#### To use GPU u should have the toolkit....
python gpu.py

## To create dataset Run :
#### Create a large dataset as possible which is helpful for better pridiction and good accuracy..
python data.py

## To Train and Extract the model Run :
python train.py

## To predict the realtime face liveness
python prediction.py
