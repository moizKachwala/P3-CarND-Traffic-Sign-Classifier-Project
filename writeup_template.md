# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./trainset_labels_histogram.png "Training Set"
[image2]: ./valid_labels_histogram.png "Validation Set"
[image3]: ./testset_labels_histogram.png "Test Set"
[image4]: ./before_fake_data.png "Distribution Before Fake Data"
[image5]: ./after_fake_data.png "Distribution After Fake Data"
[image6]: ./images-before-grayscale.png "Images before grayscale"
[image7]: ./Gray_scale_images.png "Gray Scale Images"
[image8]: ./test_images/1.jpg "Traffic Sign 1"
[image9]: ./test_images/12.jpg "Traffic Sign 2"
[image10]: ./test_images/13.jpg "Traffic Sign 3"
[image11]: ./test_images/15.jpg "Traffic Sign 4"
[image12]: ./test_images/17.jpg "Traffic Sign 5"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/moizKachwala/P3-CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data ...

![Test Dataset][image1]
![Validation Dataset][image2]
![Test Dataset][image3]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

Here is my pipeline to process the images.

    1. First I decided to convert the images to grayscale. The images before and after gray scale is : 

![Images before Grayscale][image6]

![Images after Grayscale][image7]

    2. Then I normalize the images to center the data around zero mean.

    3. Then I decided to generate some fake data. I rotate -/+ 15 deg of the images and saved them to my training set.

    Here is the histogram for the training set before and after adding fake data.

![Distribution before adding Fake Data][image4]

![Distribution after adding Fake Data][image5]



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   					| 
| Convolution        	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride, Input = 28x28x6. Output = 14x14x6 |
| Convolution   	    | Output = 10x10x16      						|
| RELU          		|           									|
| Max pooling			| Input = 10x10x16. Output = 5x5x16             |
| Fully Connected		| Input = 400. Output = 250.                    |
| Dropout layer			|												|
| Fully Connected		| Input = 250. Output = 160						|
| RELU          		|           									|
| Fully Connected		| Input = 160. Output = 84						|
| RELU          		|           									|
| Fully Connected		| Input = 84. Output = 43						|


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The configuration that is used to train the model is at cell 16. It consists of a number of training epochs and a batch size. The number of epochs defines how many times the model is trained with the whole set of training samples after adding fake data. Some tests with a high number of epochs showed that the model did not get any better after ~20 epochs but started to overfit. Also an increase of the batch size (which was only possible on another machine because of memory reasons) didn't show a good effect. Using a big batch-size resulted in hitting some local minimum, which forced the overall performance to stay at a medium level. The selected values lead to the best result for the chosen network and training set.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of 0.990
* validation set accuracy of 0.971
* test set accuracy of 0.951

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image8] ![alt text][image9] ![alt text][image10] 
![alt text][image11] ![alt text][image12]

I picked up from the wikipedia.org and resized them to fit the training and validation data. I think this would be easy to predict as the image quality is very good. If we have low quality image then it might be that the prediction fail. There are other reasons where it can fail is the lighting conditions on the images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Unfortunately all images have been detected correctly, so there is not much to discuss about the performance of the recognizer.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Speed Limit (30km/h)  | Speed Limit (30km/h)								        | 
| Priority Road			| Priority Road						|
| Yield					| Yield											|
| No Vehicle	      	| No Vehicle					 				|
| No Entry		    	| No Entry      							|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%. There were some of the prediction that got .92% and 0.95% of the accuracy and some got 100% accuracy.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 21th cell of the Ipython notebook. The main reason I see is the good quality of the image. Also it could be the reason where we have the exact same image in the training set and our model has learned.

There are other images whose accuracy is not 100% and I think the reason could be the model has overfit.

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| 1.0         			| Speed Limit (30km/h)   									| 
| 1.0     				| Priority Road 										|
| 1.0					| Yield											|
| 1.0	      			| No Vehicle					 				|
| 1.0				    | No Entry      							|


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?


