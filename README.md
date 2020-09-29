# AI Programming with Python Project

Project code for Udacity's AI Programming with Python Nanodegree program. In this project, I first developed code for an image classifier built with PyTorch in Jupyter Notebooks, then converted it into a command line application. An HTML version of the notebook can be found [here](https://github.com/c-mcdonnell/image_classifier/blob/master/Image%20Classifier%20Project.html), the Jupyter notebook format can be found [here](https://github.com/c-mcdonnell/image_classifier/blob/master/Image%20Classifier%20Project.ipynb)  and the two command line scripts can be found [here](https://github.com/c-mcdonnell/image_classifier/blob/master/train.py) and [here](https://github.com/c-mcdonnell/image_classifier/blob/master/train.py).

The image classifier was trained on [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories using backpropagation and a vgg16 network that was trained on the ImageNet dataset to get features. 

`Train.py` builds an untrained feed-forward network as a classifier using ReLU activation and dropout, trains this classifier using backpropagation, and then validates the network and tests it on data it has not yet seen. Over 5 epochs, it returns testing accuracy upwards of 80%. It then saves a checkpoint that can be used to rebuild the model in other scenarios.

`Predict.py` takes an image and a model as inputs and returns a prediction for the image's class. It loads the checkpoint that is saved in Train.py, preprocesses the image to comply with the correct size and color channel used for the model. It then predicts the top 5 most probable classes and returns the class with the highest probability.
