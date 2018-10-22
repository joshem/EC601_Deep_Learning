# Tensor Flow Implementation
This project was done for a course, this one geared towards Machine Learning.  These classes were separated into Trains and Cars, specifically. Note however, more categories can be added.

For installation, please follow instructions on https://www.tensorflow.org/install/pip (this installation uses pip, which it assumes you have not installed yet).  Note that for this design, you will also need tensorflow_hub.  

There are a multitude of dataset sites offered online.  I used Kaggle, although there are plenty of ways to use this.  To install, follow installations instructions on https://github.com/Kaggle/kaggle-api.  You will need to export your username and key.  You must create a Kaggle account, and request an API key if you don't already have one.
After this, you can go to Kaggle and search for datasets. I used the famous Stanford cars dataset.  I also used a trains dataset. To download the dataset, have a terminal open to the directory you prefer, and copy the clipboard command that should be on the Kaggle site for the dataset you want.

In order to implement my ML design, I used Tensor Flow licensed programming, and have properly cited as such.  The tf_retrain.py is based on the retrain.py found here: 'https://github.com/tensorflow/hub/blob/master/examples/image_retraining/retrain.py/'.  This trains a model to recognize against multiple classes, in which training data is pre-sorted into directories within the working directory.  For my design, I had a cars directory and train directory in the same folder as tf_retrain.py.  
Note that retrain takes a significant amount of time, so I significantly reduced the sample size of the Stanford dataset to only 2000 pictures.  The dataset of trains only had ~1000 images.
To run retrain, run the following:
python3 tf_retrain.py --image_dir /PATH/TO/MY/TRAINING/DATA

The tf_label_image.py tests an image, giving confidence of the result. To run, do the following:

python tf_label_image.py \
--graph=/tmp/output_graph.pb --labels=/tmp/output_labels.txt \
--input_layer=Placeholder \
--output_layer=final_result \
--image=/PATH/TO/TESTING/TEST.jpg

 
Running these tests, with both test files, here are the results:

testCar.jpg:
cars 0.99988115
train 0.00011882853

testCarBlurry.jpg
cars 0.74984735
train 0.25015265

testTrain.jpg:
train 0.9996586
cars 0.00034138633

testTrainBlurry.jpg:
train 0.9999548
cars 4.515491e-05

One may refer to the pictures in this repo to see for oneself how well these results work.  However, I will state that the results are accurate, never falsely identifying in these particular trials. 

This seems to be an elegant design, but there are some issues of note.  Mainly, the tmp folder gets cleared frequently, so the learning data could be cleared, meaning that there will have to be retraining.  Improvements would be to tweak this dense code such that it outputs to a more convenient location.

# ImageAI Implementation
ImageAI is another implementation of object detection, based off of Tensorflow as well as OpenCV.  The creators are experienced within the field, also creating an extensible training framework for PyTorch.  You can see their work for ImageAI here: https://github.com/OlafenwaMoses/ImageAI/
I recommend going to their site for full instructions, as there are a lot of dependencies, but I will give you a list of dependencies as of writing this:

pip3 install --upgrade tensorflow

pip3 install numpy

pip3 install scipy

pip3 install opencv-python

pip3 install pillow

pip3 install matplotlib

pip3 install h5py

pip3 install keras

pip3 install https://github.com/OlafenwaMoses/ImageAI/releases/download/2.0.2/imageai-2.0.2-py3-none-any.whl 

I also had to download tkinter: sudo apt install python3-tk

There are various different implementations for this model, listed in ImageAI_simple and ImageAI_complex.  Both use Resnet, a deep learning model developed by Facebook. This image detector allows for the speed of single stage detectors, while surpassing the accuracy of double-stage detectors. The more complex program also requires YOLOv3, a similar detector.  The download links for these are in the github for ImageAI (note that they are very large).

You can run the program with:

python3 ImageAI_simple.py 

OR

python3 ImageAI_complex.py

# System Comparison
## TensorFlow
TensorFlow is a very popular machine learning tool developed by Google.  It is very well documented, and is constantly getting updated and improved upon.  It is very extensive, however, the documentation is equally as intensive, and there are great tutorials, both from Google and from third party groups.  It uses Keras, which is a fantastic API in order to create neural networks.  TensorFlow requires that you teach your machine, meaning that you must create your own models.  However the documentation to do this is very extensive, and some programs to do this are even provided.
## ImageAI
ImageAI is an extension upon TensorFlow and OpenCV, among other projects. It was built to support all machine learning frameworks under the sun, and be incredibly easy to use, using as few lines of code as possible.  I think they succeeded in this regard, as both programs are under 20 lines of code.  Most lines are API calls to other ML algorithms. The support for RetinaNet means that you can identify up to 80 different objects, without having to train a single thing.  It also supports the ability to create your own models within its framework. This seems like a really well thought out, extensive library that seems very useful in order to create an easy to deploy machine learning framework. 
