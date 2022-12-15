# ReCAPTCHA
Use a CNN to classify dog and cat images

Dataset Description:

The dataset I used was from Kaggle (you can download the data here: https://www.kaggle.com/competitions/dogs-vs-cats/data), and it contains 25,000 images of cats and dogs (12,500 of each type).  Training and Test data were both provided for modeling purposes.  
Web services are frequently protected with a test that's supposed to be easy for people to solve but difficult for computers. These challenges are often called CAPTCHA or HIPs, which are used to reduce email and spam, and prevent attacks on passwords.  Asirra is a HIP that works by asking users to identify photographs of cats and dogs.  
Asirra partners with Petfinder.com, and they provided Microsoft Research with over three million images of cats and dogs that were manually classified by people from animal shelters across the United States.  These data were provided to challenge users to train and test a machine learning classification model to identify vulnerabilities in CAPTCHAs and HIPs. 
P
roblem:
There is wide assortment of photos in imaging databases, which makes accurate classification using deep learning rather difficult.  Past polls from computer vision experts theorized that a classification model better than 60% accuracy would be difficult without a major advance in computer technology.  A 60% classifier alone improves the guessing probability of a 12-image HIP from 1/4096 to 1/459, a major increase in the probability of making a better guess (i.e. much better than random).  Image data is vulnerable to attacks when accuracy of machine learning classification performs at or above 80%, and literature suggests this is now possible as technology has improved over the years. 
It is believed that image recognition attacks are becoming more prominent as technology advances and allows users to use deep machine learning to make better than random guesses.  By better understanding how the data can be attacked, Asirra can update their processes to better protect their data from such vulnerabilities.
Can I perform better than 80% accuracy?  Can I use machine learning to discern images of dogs from images of cats and beat the CAPTCHA (Completely Automated Public Turing test to tell Computers and Humans Apart) or HIP (Human Interactive Proof) challenges in place to protect Asirra (Animal Species Image Recognition for Restricting Access)?  Can I use deep learning to crack the Asirra CAPTCHA/HIP?  

Model:

This model is developed to learn the distinguishing features between a cat and a dog.  I am going to use Kernels and KERAS to create a neural network to identify images of cats and images of dogs.  Specifically, I will use a convolutional neural network (CNN) to classify these images and train my algorithm to predict the labels for the test data (1 = dog, 0 = cat). 

Why a CNN?
1.	Rugged to distortions in images (different lighting, poses, shifts, occlusions, etc.)
2.	Doesn’t take a lot of memory (uses same coefficients across  different locations in space)
3.	Easier and better training than standard (reduction in parameters, less noise in training process) 

Analysis (coding):

Import the libraries and define the constants:
 
Prepare the training data and look at the data:
  
Build the model: 
 
Early Stop prevents over-fitting, so it’s set up to stop the learning after 10 epochs and when val_loss value is not decreased
Learning Rate Reduction is reduced when then accuracy does not increase for 2 steps

Prepare Data:
Since I use image generator with class_mode="categorical", I need to convert the column category into a string.  The image generator will then convert it one-hot encoding, which is good for classification.
  
Fit and Save the model:
 
The model seems to be pretty accurate.
 
Virtualize the training:
  
Prepare the test data, run the test generator, and predict:
 
Virtualize the results and look at a sample of the predictions:
 
Export the data and explore the output:
 
Success!

Difficulties:

I ran into a few issues with running these data:

1.	Took me some time to learn the nuances of Spyder having only worked with Jupyter in the past
2.	Errors in making changes to directories with coding supplied
3.	Length of time to run (large dataset, large epochs, etc.)
a)	RAM processing might have been an issue with computer
4.	Inability to run single cells to test code (found out selecting cells and F9 ran selected cells afterward running the entire program)
5.	Not fully understanding how to set epochs and dropouts for such a large dataset
6.	Installing some environments was difficult:
a)	I had to use Anaconda Prompt to install some libraries because I would get errors 

Next Steps:

Although this model has relatively high accuracy and includes a large amount of data, there are some ways to improve upon the model:
1.	Get more data
2.	Try new/different model architecture
3.	Decrease number of features 
4.	Introduce regularization such as the L2 regularization.
5.	Make the network shallower (less layers)
6.	Use a smaller number of hidden units
7.	Supplement using cross-validation methodologies
8.	Use a pre-trained network (i.e. transfer learning)
9.	Tune some hyperparameters:
a)	epochs, 
b)	dropouts
c)	learning rates, 
d)	input size, 
e)	network depth, 
f)	backpropagation algorithms, 
g)	GPU batch size 
Big picture next steps might also include:
1.	Apply this program to more difficult image recognition projects 
2.	Research methods of improving upon image recognition safeguards

Conclusions:
With an accuracy of nearly 90%, I find that there is a machine learning image recognition process is applicable with current technology that can reach >80% accuracy in identifying images of cats and dogs.  Asirra appears to have some vulnerabilities that need to be addressed before being deployed securely; however, Asirra can likely be applied safely and securely by using additional protections.  One research paper addresses a token bucket scheme that would provide such a safeguard that might allow for the safe implementation of Asirra.  
[Machine Learning Project Paper_Susannah Craig.pdf](https://github.com/skcrai2/ReCAPTCHA/files/10237363/Machine.Learning.Project.Paper_Susannah.Craig.pdf)
