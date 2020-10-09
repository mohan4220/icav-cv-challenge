# icav-cv-challenge

I am using a RCNN type model that uses selective search for regional proposals.

you can download the dataset from here: 

I created the pickles files using main.py script, of the training images and lables. you can download them from here:


My train.py script uses pickled image and label files to train the model. model architecture is also defined in this file.

in order to predict, use predict.py file as:
python predict.py <path to image file>
  
Its a pretty basic model, as it uses selctive search at every stage to have regional proposes, it is slow. performance and accuracy can be increase by using a model that has region proposal mechanism build into the network like a faster RCNN.
