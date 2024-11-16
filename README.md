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

## Audio
pip install opencv-python tensorflow numpy speechrecognition pyaudio



# Create a virtual environment
python -m venv venv
venv\Scripts\activate


# To create dataset Run :
python data.py

# To Train and Extract the model Run :
python train.py

# To predict the realtime face liveness
python prediction.py
