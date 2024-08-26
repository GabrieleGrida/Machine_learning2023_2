# Wild Cat Image Classification with Neural Networks: FFNN vs. CNN
How to Implement a fully connected feed forward neural network and a convolutional neural network to classify images from a dataset, in this case I refer to the "Cats of the Wild" dataset.

Task 1: Image Classification with Fully Connected Feed Forward Neural 
Networks (FFNN)
In this task, I will try and build a classifier for the provided dataset, using a classic Feed 
Forward Neural Network.
1. First of all, I need to download the dataset, which consists of 7 classes with a folder for 
each class image. In particular, the classes are: 'CHEETAH', 'OCELOT', 'SNOW 
LEOPARD', 'CARACAL', 'LIONS', 'PUMA', 'TIGER'. We define the paths to the image 
dataset and the “utils” file, containing some useful implemented functions that we will use to 
load the data and save models. Then, we add them to the system path so that these directories 
can be accessed by Python to import modules or read files. After that, we import all functions 
from the “utils” file, so that we can use them to load the images and make a dataset with the 
respective functions. In this way, we obtain two arrays: “x”, which contains the images, and 
“y” which includes the corresponding labels in a numerical form. We can see the shapes of x 
and y, which are (1620, 224, 224) and (1620, 7) respectively. Moreover, we display a sample 
of images with their labels using the ‘plot_sample’ function. This step is present in the first 
cell, which must necessarily be run for the rest of the code to work. 
2. In the second cell I start to work on building the first model. First of all, I import all the 
libraries and tools that I will use later, therefore you need to install them if necessary to 
run the code. Then I set a random seed to reproduce the results of this script and I split 
data into a training and a test set using indices. Moreover, I need to normalize the image 
data by dividing each pixel value by 255, given that the range for an image pixel is 0-255
and we want the range to be [0, 1]
3. After that, we reshape data into 1D vectors for processing by the neural network and we 
set the number of classes, which is 7 in this case. Note that the labels are already 
expressed in binary form, so we do not need to use a one-hot encoding.
4. Now, we can build the model that will solve the classification task using the Keras 
‘sequential’ model. In particular, it is a simple Feed Forward Neural Network composed 
of three dense layers: two hidden layers each with 128 neurons, and an output layer of 
size equal to the number of classes, 7 in this case. The first one is also the input layer as it 
specifies the number of inputs to this layer, using the ‘input_shape’ argument. 
Moreover, the activation functions used are ‘relu’ for the hidden layers and ‘Softmax’ for 
the output layer, as we are solving a multi-class classification task. In the next step, I 
compile the model using the Adam optimizer with a learning rate of 0.001, the 
categorical cross entropy as a loss function, and the accuracy as the metric. Before fitting 
the model, we also define an early stopping to prevent overfitting. Then, we can train the 
model using the training set with a batch size of 32 for 8 epochs. Moreover, we set a 
validation split of 0.15. We save and load the model using the implemented function 
imported from “utils”, and we evaluate the model on the test set. In this case, we obtain a test 
loss of 2.465 and a test accuracy of 0.247, which is quite low.
5. We can plot the training and validation accuracy over epochs using the ‘plot_history()’ 
function from “utils”. We can see that training accuracy is generally higher than the 
validation accuracy, since the data are biased from the fact that they were used to train the 
model. Therefore, the accuracy computed on the validation set is more reliable as it relies on 
unseen data. The plot also marks the best epoch, the one with the lowest validation loss, 
which is the 7th in this case.
6. Since we obtain a test loss of 2.465 and a test accuracy of 0.247, we can say that the 
model is performing poorly on the test set. Indeed, it is predicting the correct class only 
24.7% of the time approximately. It is quite low, so we do not expect a high classification 
accuracy when it makes predictions on new and unseen images. Therefore, we need to 
improve the model.
7. In the third cell, we build another model with a different architecture. Indeed, we do not 
directly use the flattened images as input, but we extract some features from each image 
and we use them for training and testing the model. In particular, I will build two models 
using different types of features and I will evaluate each of them to save the best one. In 
the first model, I extract features from the whole image. The features computed for each 
image are: the mean, which is the average pixel value over the entire image; the variance, 
which measures the pixel value dispersion; the minimum and the maximum pix values.
We compute these features for both the training and the test set and we stack them into a 
single array for each set. Therefore, we use these arrays as input to train and test the 
model instead of the original images. Fitting the model on these data, without changing 
other parameters, we evaluate the test set and we obtain a test loss of 1.897 and a test
accuracy of 0.191. In this case, the model accuracy is lower than before. This could be 
explained by the fact that when you flatten an image, you retain all the original pixel 
data, while here we are extracting specific features reducing significantly the amount of 
data used to train the model. As a result, the performance of the model trained on these 
features is lower than that trained on the flattened images. Therefore, we can try to 
extract other features, for example from each channel. Indeed, there are usually three 
channels for RGB images: one for red, one for green, and one for blue. In case, we 
compute the mean, variance, min, and max for each color channel separately, and the 
ratio between the maximum value of different channels (e.g. max red/max blue). As 
before, we compute these features for the training and test set, stack them into single 
arrays, and use them to train and evaluate the model. This time, we have a test loss of 
1.77 and a test accuracy of 0.318. This result indicates that the features extracted in this 
case are providing more useful information for the classification task, compared to the 
previous ones. Therefore, we can save this last model as the best one.

Task 2: Image Classification with Convolutional Neural Networks (CNN)
In the second task, I must implement a multi-class classifier (CNN model) to identify the
class of the images.

1. The first steps are the same as in task 1. Therefore, once we downloaded the data in the first 
cell, we need to prepare data splitting indices into training and non training set. The nontraining set is, in turn, divided into a test and validation set. Then we normalize data to 
obtain values in a range of [0, 1].
2. Now, we define a CNN model using the ‘Sequential()’ function. This is a type of neural 
network typically used for image-processing tasks. In this case, it is composed of three 
convolutional layers, which evaluate affinities based on the principle of locality. Each layer 
is made up of a large number of filters, or kernels, that are used to detect the presence of 
local features in an image. Then there are three pooling layers (two max pooling, one 
average pooling), which reduce the dimensionality of the feature map in order to decrease 
processing time. Since the channel feature maps are reduced in size after each pooling layer, 
the number of filters increases in the next convolutional layer to compensate (16, 32, 64), 
while the kernel size remains the same (3x3). These layers are followed by a flatten layer 
that transforms the 2D outputs into 1D. Finally, there are two fully connected (dense) layers 
and the output layer. The latter uses the ‘softmax’ activation function as before, given that is 
suitable for multi-class classification tasks. 
3. Then, we compile the model setting the Adam optimizer with a learning rate of 0.01, the 
categorical cross entropy as the loss function since it is suitable for multi-class classification, 
and the accuracy as the metric. Now, we can train the model using a batch size of 32 and 8 
epochs. Moreover, we shuffle data before each epoch, setting shuffle=True. We use early 
stopping to stop the training if the validation loss does not improve, and we set validation 
data used to evaluate the model after each epoch.
4. Plotting the training and validation accuracy, we can observe that they follow a similar trend 
as before since training accuracy increases more than validation accuracy. However, both are 
on higher levels of accuracy than before. Indeed, if we evaluate the performance of the 
trained model on the test set, we obtain the following results: test loss = 2.023; test accuracy 
= 0.486. The accuracy has almost doubled compared to the previous model where we used a 
simple Feed Forward Neural Network. However, even if we observe an improvement also in 
the test loss, from 2.465 to 2.023, the accuracy is still too low since the model is correct less 
than 50% of the time. Therefore, we can conclude that the CNN model is more successful in 
solving this classification problem, given the significant improvement in the test loss and 
accuracy. However, we can still improve the model making it perform better.
5. Qualitatively, the performance of the two models is different because of their different 
architecture. Indeed, in T2 we built a Convolutional Neural Network (CNN), which handles 
image data more efficiently as it considers the spatial structure of the image, recognizing 
patterns over the image through the application of filters. Whereas the model in T1 is a dense 
Feed Forward Neural Network (FFNN), which treats each pixel independently, ignoring 
spatial dependencies and hierarchical patterns within the images. Statistically, we can see 
from the evaluation of the performance on the test set that the CNN model performs better 
than the FFNN model in terms of accuracy. Indeed, CNN achieved a test accuracy of 0.486, 
which is significantly better than the test accuracy of the other model (0.247).
6. One method to improve the model consists in using manipulation/augmentation techniques 
on data to get better performance. In particular, using these techniques we significantly 
increase the diversity of the data available for training the model, without collecting new 
data. We can manipulate the images using Keras’ util ImageDataGenerator().
In this case, I chose five different types of manipulation: random changes in the intensities of 
the RGB color channels (‘channel_shift_range’); random rotation of images in a range of -
20 and 20 degrees (‘rotation_range’); random zooms in the image in and out by 10% 
(‘zoom_range'); random shear transformation (‘shear_range’); random horizontal flips of 
the image (‘horizontal_flips’). Therefore, we generate synthetic data for both the training
and the validation set and we use them to fit and evaluate the previous model. We maintain 
the same batch size, number of epochs, and early stopping, but we use the “new” data. 
Evaluating the performance of the model and plotting the accuracy, we can see a significant 
improvement in the results. Indeed, we obtain a test loss of 0.9581 and a test accuracy of 
0.658. We can conclude that introducing manipulation/augmentation techniques and 
increasing the diversity of the training data, can help the model generalize better and 
improve the performance on the test set.
7. Finally, we can tune the model hyperparameters to improve the performance of the previous 
CNN model. The goal is to find the best set of hyperparameters that lead to the highest test 
accuracy. In this case, we consider as hyperparameters the number of epochs and the 
learning rate. However, they could be more, such as the batch size, the number of 
convolutional layers, or the activation functions. Firstly, we define a function to create the 
model in which we will change the hyperparameters. In particular, it takes the ‘optimizer’
argument as an input to set the learning rate of the Adam optimizer. Then, we set the list of 
potential values for the hyperparameters. In this case, we can choose between a model with 
8, 10, or 12 epochs, and a learning rate of 0.001 and 0.01 for the Adam optimizer. We 
initialize the variables for the best score and the best parameters, then we can set two for 
loops to create a model using each possible combination of parameters values given, train the 
model, and evaluate its performance. At the end of each loop, the variables containing the 
optimal parameters are updated if the accuracy improves using the current model. Finally, 
we obtain the best combination of values for the hyperparameters considered. In this case, 
we can see that the highest accuracy can be reached using 12 epochs and a learning rate of 
0.001. In particular, the best score is approximately 0.58, which is higher than the previous 
CNN model (0.486), given that we are using more epochs than before. However, we 
achieved a better performance using data augmentation, so we could change the combination 
of values or modify the hyperparameters considered to obtain a higher test accuracy for this 
model.
