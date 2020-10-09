# icav-cv-challenge

I am using a RCNN type model that uses selective search for regional proposals.

you can download the dataset from here: 
https://drive.google.com/drive/folders/1T4iwoUNFdEGKly7qzFrtvjrNxlbFpkSh?usp=sharing

I created the pickles files using main.py script, of the training images and lables. you can download them from here:
https://drive.google.com/file/d/1XEdy2-d92SlITX-Ijulujsoc_MuojFSm/view?usp=sharing
https://drive.google.com/file/d/1YA6_YsuSexD3uSLSR_YWeayVo1aQdxvH/view?usp=sharing

My train.py script uses pickled image and label files to train the model. model architecture is also defined in this file.

in order to predict, use predict.py file as:
python predict.py <path to image file>

#Answers to the questions given:
1.	By changing input number of epochs, changing activation functions, choosing a suitable optimizer and loss functions.
2.	By changing number of epochs and batch size.
3.	I use tensorflow. By using tensorflow’s model optimizing toolkit.
4.	Using MLQ module at the backend.
5.	Provide more data for training, data augmentation, increased number of epochs.
6.	Overfitting can be avoided by having a large dataset with unsimilar items, reducing the number of epochs.
7.	I try to split the dataset, then try to create another dataset from the given hoping to make it balanced.
8.	I feed the live image/video feed from a camera to the model, make predictions and output the results, then pass it to another node or code segment to decide what to do with the results.

Its a pretty basic model, as it uses selctive search at every stage to have regional proposes, it is slow. performance and accuracy can be increase by using a model that has region proposal mechanism build into the network like a faster RCNN.
