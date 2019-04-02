# **Self Driving Car Engineer**
---
## Project III: Traffic Sign Recognition

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train and validate a model so it can classify traffic sign images using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then try out your model on images of German traffic signs that you find on the web. The code for this project is located in an IPython notebook in ./Traffic_Sign_Classifier.ipynb.


This writeup will address the following [rubric points](https://review.udacity.com/#!/rubrics/481/view)

* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./output_images/sign_examples.png "random examples"
[image2]: ./output_images/sign_hist.png "sign distribution"
[image3]: ./output_images/new_sign_images.png "images downloaded from web"
[image4]: ./output_images/processed_images.png "pre-processed image"
---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the methods len() and shape() to calculate the number of examples and to determine the shape of the images. Here is a short summary of the traffic sign dataset:

* Number of training examples = 34799
* Number of testing examples = 12630
* Number of validation examples = 4410
* Image data shape = (32, 32, 3)
* Number of classes = 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. First, I plotted 12 random examples from the dataset shown below:

![alt text][image1]

Second, the following histogram shows the traffic sign distribution:

![alt text][image2]


### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

All datasets were converted to grayscale and normalized between -1 and 1 with the following formula:

```
(X - 128) / 128
```
Converting to grayscale improved the accuracy in the Sermanet and LeCun publication about traffic sign classification, and was thus used here as well.

The data normalization makes all features weighted equally. In general, normalization rescales values to fit in a specific range to assure better convergence during backpropagation.

As the last step the pre-processed dataset was shuffled.

![alt text][image4]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

I adapted the LeNet model and included dropout for better regularization. My final model consisted of the following layers:

| Layer         		    |     Description	        					            |
|:---------------------:|:---------------------------------------------:|
| Input         		    | 32x32x1 grayscale image   							      |
| Convolution 1       	| 1x1 stride, VALID padding, output = 28x28x6 	|
| RELU					        |												                        |
| Max pooling	      	  | 2x2 stride, VALID padding, output = 14x14x6 	|
| Convolution 2	        | 1x1 stride, VALID padding, output = 10x10x16	|
| RELU		              |        								                       	|
| Max pooling			    	| 2x2 stride, VALID padding, output = 5x5x16		|
| Flatten   						|	output = 400              										|
|	Fully connected				|	input = 400, output = 120											|
| RELU                  |                                               |
| Dropout               |                                               |
|	Fully connected				|	input = 120, output = 84											|
| RELU                  |                                               |
| Dropout               |                                               |
| Fully connected	      | input = 84, output = 10                       |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

I started with a pre-defined LeNet architecture, which was introduced in the Udacity lecture. I changed the model by modifying the number of output neurons according to the number of output classes.

To compute the cost function of the model I used `tf.softmax_cross_entropy_with_logits` together with `tf.reduce_mean`. The cost function was then minimized using AdamOptimizer.

First, I tried to train the model with only normalized gray-scaled images. To improve the accuracy of the model I added dropout after activation function of each fully-connected layer. I tweaked the parameters until I reached a test accuracy of > 0.93.

I trained the final model with 20 epochs, a batch size of 128, a learning rate of 0.001 and a dropout keep probability of 0.65


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.


As mentioned above I started with a pre-defined LeNet architectures, which was introduced in the Udacity lecture. In order to increase my accuracy I used and iterative approach. During each iteration I changed either the model by adding dropout or the hyperparameters. Dropout was incorporated into the model as a regularization to prevent overfitting. The changes and the final accuracy or each iteration were carefully noted.

Each iteration was evaluated on the validation set. When the performance was satisfactory I evaluated the model on the test dataset to see how the model performs on new images. My model performed > 0.93 on the test.

My final model results were:
* validation accuracy -> 0.941
* test set accuracy -> 0.934

To get higher accuracy one would have to get more training examples, especially for the underrepresented classes. This could for examples be achieved by augmenting the data or by using generative adversarial network to generate new images.

A better method to tune the hyperparameters would be to implement a grid search or a similar method.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are 8 German traffic signs that I found on the web:

![alt text][image3]

In comparison to the original dataset, the web images seem to me to be somewhat sharper. Meaning, the transition between edges is not so smooth as in the original dataset. This might play a role in the classification.


#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Road work | Road work |
| General caution | General caution |
| Right-of-way at the next intersection | Right-of-way at the next intersection |
| Speed limit (30km/h) | Speed limit (30km/h) |
| No vehicles | No vehicles |
| No entry | Priority road |
| Speed limit (50km/h) | Stop |
| Roundabout mandatory | Roundabout mandatory |

The model was able to correctly guess 6 of the 8 traffic signs, which gives an accuracy of 75%. In comparison to the test dataset accuracy of > 93% the model performance dropped down significantly. One of the possible reasons for this could be that the image quality is different from the original dataset (see above for more details). Another reason for the false classification could be that the images from the web have different background. Hence, these images could deviate from what the classier was trained for. However, it is intriguing that the classifier did not recognize signs with a pretty good sample representation in the dataset. To determine why these images were not recognize would require more in depth comparison of the training samples.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

On 5 out of 8 images the model predicted the images with > 99% accuracy (Image ids: 25, 18, 11, 12, 40).

ID 15 - no vehicles - was predicted correctly but with a rather low accuracy of only 23.15%, followed by speed limit (50km/h) with 20.77%. Both signs look very similar, with speed limit having the additional number on a white background. Therefore, it is not surprising that the classifier assigned speed limit class the second highest probability. Furthermore, this is also enforced by the lower number of training examples for the 'no vehicles' class.

ID: 2 - speed limit (50km/h): I have no explanation how speed limit could have gotten wrongly assigned to a stop sign witch such a high probability. Both sign classes look very different and I don't see what they could have in common.

ID: 17 - No entry: Same as for ID 2. At least here 'No entry' sign class achieved a probability of 11.26%.

---

ID: 25 ==> Road work

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| Road work | 100.00% |
| Beware of ice/snow| 0.00% |
| Road narrows on the right| 0.00% |
| Bumpy road | 0.00% |
| Bicycles crossing | 0.00% |


ID: 18 ==> General caution:

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| General caution | 99.24% |
| Traffic signals | 0.76% |
| Pedestrians | 0.00% |
| Road narrows on the right | 0.00% |
| Keep right | 0.00% |


ID: 11 ==> Right-of-way at the next intersection:

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| Right-of-way at the next intersection | 99.99%
| Priority road | 0.00%
| Double curve | 0.00%
| Beware of ice/snow | 0.00%
| Roundabout mandatory | 0.00%


ID: 1 ==> Speed limit (30km/h):

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| Speed limit (30km/h)| 100.00% |
| Speed limit (20km/h) | 0.00% |
| Stop | 0.00% |
| Speed limit (70km/h) | 0.00% |
| General caution | 0.00% |


ID: 15 ==> No vehicles:

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| No vehicles | 23.15% |
| Speed limit (50km/h)| 20.77% |
| Priority road | 18.48% |
| Speed limit (30km/h) | 13.43% |
| Roundabout mandatory| 9.70% |


ID: 17 ==> No entry:

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| Priority road| 66.61% |
| Yield | 21.51% |
| No entry | 11.26% |
| Turn left ahead | 0.45% |
| No passing | 0.05% |


ID: 2 ==> Speed limit (50km/h):

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
| Stop | 99.10% |
| Speed limit (60km/h) | 0.77% |
| Speed limit (30km/h) | 0.09% |
| Keep right | 0.04% |
| Speed limit (50km/h) | 0.01% |


ID: 40 ==> Roundabout mandatory:

| Prediction         	|     Probability	 |
|:---------------------:|:--------------:|
|Roundabout mandatory | 100.00% |
|Right-of-way at the next intersection | 0.00% |
|Vehicles over 3.5 metric tons prohibited | 0.00% |
|Priority road | 0.00% |
| Pedestrians | 0.00% |


### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)
#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
