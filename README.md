# DIABETES-RETINOPATHY-DETECTION-USING-MACHINE-LAEARING

### ABSTRACT

The health care industry produces a massive amount of data. This data is not always made use to the full extent and is often underutilized. Applying predictive analysis on these datasets, a disease can be detected, predicted, and help in getting treatment at an early stage. A considerable threat to humankind is caused by conditions like cancer, diabetes, or even brain tumor. Diabetic retinopathy (DR) is an extreme eye infection which happens because of diabetes mellitus, and it has developed as the most widely recognized reason for visual deficiency on the planet. The patient’s vision can be affected by diabetes, which causes cataracts, glaucoma, and, most importantly, damage to blood vessels inside the eye, which can lead to complete loss of vision. There are effective treatments for diabetic retinopathy, but this requires early diagnosis and continuous monitoring of patients with diabetes. Diabetic retinopathy is diagnosed by assessment of retinal images. Since manual diagnosis of these image scans is slow and resource-demanding to determine the severity of DR, for the accurate detection of Diabetic Retinopathy, an efficient machine learning technique is to be developed, which extracts essential features of the retinal scan further to classify the image to different Diabetic Retinopathy severity levels. This system focuses on applying classification and prediction models on retinal images to detect diabetic retinopathy attributes for effective monitoring of the patient’s health conditions.

### OBJECTIVE

•	To study the present Diabetic Retinopathy disease prediction trends and collect data. 
•	To implement different technical indicators. 
•	To perform pre-processing on the dataset gathered to enhance the important features for detecting DR disease. 
•	To build prediction model using different machine learning models. We will combine the technical indicators with dataset and then compare the model accuracy. 
•	Build a Machine Learning model to predict the level of severity of DR disease.
•	To find significant features by applying machine learning techniques.
•	To compare different ML models and analyse which model performs more efficiently. 
•	To overall assist patients and doctors to help detect early signs of DR disease. 


### MOTIVATION

The motive behind choosing this project was to use our engineering knowledge to better the world. There is a tremendous amount of machine learning applications in the healthcare industry. The healthcare industry generates a vast amount of data, which is oftentimes underutilized. My primary motivation in choosing this project was to help the people by giving them a chance to improve their lifestyle. Hence, it was decided to make a machine learning-based healthcare project for Diabetic Retinopathy patients who require regular monitoring of their health conditions. The major motivation is that early detection can be done, which leads to initial treatments, thus increasing the chances of better recovery by regularly monitoring the eye scan.

### BACKGROUND

Diabetic retinopathy is a pervasive eye infection in diabetic patients and is the most well-known reason for visual disability/visual impairment among diabetic patients. DR, an interminable, dynamic eye malady, has ended up being one of the most widely recognized reasons for vision disability and visual impairment, particularly for working ages on the planet today. It results from delayed diabetes. Veins in the light-delicate tissue (i.e., retina) are for the most part influenced in diabetic retinopathy. The non-proliferative diabetic retinopathy (NPDR) happens when the veins release the blood in the retina. The Proliferative DR (PDR), which causes visual impairment in the patient, is the following stage to NPDR. 

The advancement of DR can be ordered into four phases: mild, moderate, serious non-proliferative diabetic retinopathy and the propelled phases of proliferative diabetic retinopathy. Figure below depicts the different stages of DR disease. 

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122230561-643de300-ced7-11eb-8f1b-c01f763aea6c.png">

In mild NPDR, little regions in the veins of the retina, called microaneurysms, swell like an inflatable. In moderate NPDR, various microaneurysms, hemorrhages, and venous beading happen, making the patients lose their capacity to ship blood to the retina. The third stage, called serious NPDR, results from the nearness of fresh blood vessels, which is brought about by the discharge of development factor. The most noticeably awful phase of DR is the proliferative diabetic retinopathy, as delineated in Figure in which delicate fresh blood vessels and scar tissue structure on the outside of the retina, improving the probability of blood releasing, prompting perpetual vision misfortune. At present, the retinopathy identification framework is cultivated by including a very much prepared doctor physically distinguishing vascular irregularities and auxiliary changes of retina in the retinal fundus pictures, which are then taken by expanding the retina utilizing vasodilating operator. Because of the manual idea of DR screening strategies, nonetheless, profoundly conflicting outcomes are found from various perusers, so robotized diabetic retinopathy determination methods are fundamental for taking care of these issues. 


In spite of the fact that DR can harm retina without demonstrating any sign at the starter stage [2], effective beginning period location of DR can limit the danger of movement to further developed phases of DR. The finding is especially hard for beginning time recognition in light of the fact that the procedure depends on perceiving the nearness of microaneurysms, retinal hemorrhages, among different highlights on the retinal fundus pictures. Besides, the exact identification and assurance of the phases of DR can incredibly improve the mediation, which at last decreases the danger of perpetual vision misfortune. Prior arrangements of robotized diabetic retinopathy location framework depended close by made component extraction and standard AI calculation for forecast [3]. These methodologies were extraordinarily enduring because of the carefully assembled nature of DR highlights extraction since include extraction in shading fundus pictures is all the more testing contrasted with the customary pictures for object location task. In addition, these hand-created highlights are exceptionally touchy to the nature of the fundus pictures, center edge, nearness of antiquities, and clamor. Along these lines, these restrictions in conventional hand-made highlights make it critical to building up a successful component extraction calculation to viably dissect the inconspicuous highlights identified with the DR location task.



### Project Description and Goals

Machine Learning (ML) plays a very significant role in almost every field today. For every industry, be it energy, healthcare, or production, it is nearing maximum automation with the help of Machine Learning techniques. Nevertheless, despite all this, the healthcare sector today faces a massive problem of unremitting monitoring from quite some time now. Not everyone can afford the costly personal healthcare devices, and not every healthcare device today consists of a check for all the vital health factors. For Diabetes Retinopathy disease, constant monitoring is a must, and early disease detection could help avoid a severe disease.  
The project aims to create an analytical system that uses medical records of patients as datasets. It performs a machine-learning algorithm to find trends and further predict the diagnosis for new patients. This project centers around choice about the nearness of sickness by applying outfit of AI characterizing calculations on highlights separated from a yield of various retinal picture preparing calculations, similar to the width of the optic plate, injury explicit, image-level. Moreover, it would help Doctors better analyze and diagnose the patients’ health issues based on the prediction and data analysis done from the healthcare dataset. The system we propose is an amalgamation of the healthcare and Machine Learning techniques.


#### PROJECT DESCRIPTION
In this project we use the well-known APTOS 2019 Blindness Detection dataset which is acquired from Kaggle, an online website which contains a large number of publicly accessible datasets. The proposed modules are:

#### 1.1	DATA COLLECTION - The dataset is taken from www.kaggle.com which contains about 3662 train images and 1928 test images, a total of 10.2GB data. The train.csv file contains the diagnosis of the train images, labelling each image ID to the severity level of the disease ranging from 0-4.( 0 being lowest and 4 being the highest severity level). The Test.csv file contains the image IDs for the test images.

Figure below depicts a snapshot of a fraction of the un processed training images. These are the raw images obtained from the dataset. As we can see, the images are not in a standard form and the lighting conditions of each image varies from the other. 

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122231519-1ecde580-ced8-11eb-8ac1-04709f0bcee3.png">

#### 1.2	DATA PRE-PROCESSING - The raw data collected has to go through a process of normalisation in order to get a uniform dataset. Thus we need to pre-process the data for accuracy. The following pre-processing steps were used in order to standardize the retina images:


* Reducing lighting-condition effects : Raw images come with many different lighting conditions, some images are very dark and difficult to visualize. Thus, we convert the images to 50% grey scale.
* Cropping Image : We cropped the image to remove the uninformative outer background.
* Resize Image : The images were rescaled to get the same radius.  
* Removing the Image Noise : Using the Gaussian Blur function in the cv2 library we reduced the image noise for a better analysis.

Figure 4 below depicts the Pre-Processed training images. The entire database has been processed by the mentioned parameters in order to standardise and enhance the attributes which determine the severity of the DR disease.

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122231791-5a68af80-ced8-11eb-8c3f-414815c2376f.png">

#### 1.3	DATA SPLITTING- For model training the train images was split into 2 sections which are ‘Training data’ and ‘Validation data. The classifier was trained using ‘training data set’, parameters tuned using ‘validation set’ and then the ‘test data set’ was used to test the performance of the classifier on unseen dataset. We randomise the train images and split it into a ratio of train and validate set in a ratio of 3112:550.

#### 1.4	BUILDING MODELLING - We will be using the CNN(Convolutional Neural Networks) model for this project. In recent times, most of the computer vision problems have been solved with greater accuracy with the help of modern deep learning algorithms. Convolutional Neural Networks (CNNs) have been proven to be revolutionary in different fields of computer vision such as object detection and tracking, image and medical disease classification and localization, pedestrian detection, action recognition, etc. The critical attribute of the CNN is that it extracted features in a task-dependent and automated way. In this paper, an efficient Neural Network architecture is presented for DR detection in large-scale databases. Our proposed network was designed with a multi-layer CNN architecture followed by two fully connected layers and an output layer. Among all the ML algorithms, we choose to use DenseNet for image classification and recognition because of its high accuracy. DenseNet (Densely Connected Convolutional Networks) is one of the latest neural networks for visual object recognition. We use DenseNet over CNN model because in DenseNet every layer is getting an "aggregate information" from every single going before layer. Since each layer gets highlight maps from every single going before layer, the system can be more compact and minimized. 



The following model architecture developed are mentioned below with their specifications.

#### 1.5.1	MODEL 1: The network architecture of our proposed ML Model 1 is as follows. The input layer of the network is 224 x 224 x 3. ReLU was used in all convolutional layers as the activation function for nonlinearity. All the Max-Pooling layers used have the same kernel size of 3 x 3. After the DenseNet121 convolutional Layers, the Global Average Pooling (GAP) layer was added to reduce overfitting. The final extracted local features were flattened before passing through fully connected layers. There are two completely associated layers, each having 1024 neurons. Dropout of 0.5 was included when the completely associated layers to decrease overfitting. Softmax activation function was used at the last output layer to produce normalized outputs. Since it is a lot of more regrettable to misclassify extreme NPDR or PDR as a typical eye than as moderate retinopathy, we considered this multi-class order as a regression issue thus a yield layer of one neuron was included. Additionally, we clipped the loss function value between the range of 0 and 4 since our group extends between these qualities.

Model 1 was compiled by using categorical cross-entropy loss function for single labeled classification. Adam, an adaptive learning rate optimizer was used in Model compilations.
#### 1.5.2	MODEL 2 : Network architecture of our proposed ML Model 2 is as follows. The input layer of the network is 224 x 224 x 3. After the Sequential Layer ,we used predefined DenseNet121 weights for this model. After that GAP layer was added to minimize overfitting followed by a Dropout of 0.5 was added. Sigmoid Activation function was used at the last output layer to produce normalized outputs. 
Model 2 was compiled by using binary cross-entropy loss function for single labelled classification. Adam optimiser was used during model compilation.


#### 1.5	EVALUATION MEASURES – After having trained the Model, we need to evaluate the results on the training and validation dataset. Based on the evaluation metrics, we tweak the parameters in order to make the Model efficient. A few standard exhibition measurements, for example, Accuracy, Loss and Quadratic Kappa Score have been considered for the calculation of this present model's presentation adequacy. Accuracy in the present setting would mean the level of cases accurately anticipating from among all the accessible occurrences. Loss is the level of occasions erroneously anticipated from among all the accessible examples. The Quadratic Kappa Score (k) is calculated as follows:-

<img width="258" alt="image" src="https://user-images.githubusercontent.com/45623734/122232645-05796900-ced9-11eb-99c6-7f2311210ce5.png">
 
To evaluate the performance of the models these evaluation metrics are used which will help in deciding which model gives a better prediction of the DR disease. 
Both the Models were trained for 15 Epoch Cycles, and the evaluation metrics was calculated at each cycle. The Model would be saved only when there was an increase in the Kappa Score at the end of each cycle. 



#### FUNCTIONAL REQUIREMENTS

##### PRODUCT PERSPECTIVE

* 1.	The system is run in Python Jupyter and has a user friendly GUI. 
* 2.	The system consist of many options in which having the chance of heart disease is predicted. 
* 3.	The predicted value ranges from 0 to 4, in which 0 represents normal eye, 1 represents mild NPDR, 2 represents moderate NPDR, 3 represents severe NPDR and 4 represents PDR. 
* 4.	An initial training is required for running the dataset on the machine model in order to do predictions. 
* 5.	The output we get from the system is in terms of performance graphs about the ML models, a csv file containing predictions for all the test images, so the person who uses this software should have some knowledge regarding interpreting all these outputs. 

##### PRODUCT FEATURES
The product has 2 machine learning models on which the training images are trained. Once the models have been trained, we can run the test images on them and compare the predictions. 

##### USER CHARACTERISTICS
With the help of this system any user can monitor the severity level of their DR disease and can seek for the required treatment if needed. This is of great use to patients, doctors, hospitals and health ministry of the country.

    
##### ASSUMPTION AND DEPENDENCIES
The minimum hardware and software requirements has to be met for proper     implementation of this system by the user. The user is needed to provide prior information of their eye images. The image provided is clear and attributes are visible.

##### DOMAIN REQUIREMENTS
For this project, we will use Python for writing our source code. We will use many libraries like Pandas for data manipulation and analysis, Numpy for Mathematical computation library, TensorFlow for machine learning applications, Keras for Deep learning Library, OpenCV for image processing, Scikit-learn for multiple machine learning applications and matplotlib as plotting library.

##### USER REQUIREMENTS
User should have proper information regarding used data (database), present data and the meaning of the outcome to see the changes.


### SYSTEM ARCHITECTURE

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122232959-4a050480-ced9-11eb-8126-78043ccbb0c9.png">

### SYSTEM USECASE DIAGRAM

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122233019-55f0c680-ced9-11eb-9101-9374bd6837f7.png">



### Project Demonstration

* Sample of Database

![image](https://user-images.githubusercontent.com/45623734/122233237-86386500-ced9-11eb-987a-17f41986fc48.jpeg)

* Dataset Info

<img width="349" alt="image" src="https://user-images.githubusercontent.com/45623734/122233325-96e8db00-ced9-11eb-9fb9-be17fc7cee3c.png">

* Data Pre-Processing

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122233455-b41da980-ced9-11eb-8694-80d019d5b87f.png">

* Machine Learning Model 1 Training

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122233507-c0096b80-ced9-11eb-94c2-5e5d693954ad.png">
<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122233531-c7307980-ced9-11eb-8685-81e826e7998d.png">

* Machine Learning Model 1 Evaluation Metrics

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122233617-d6afc280-ced9-11eb-8df8-736823b702b1.png">
<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122233628-dadbe000-ced9-11eb-8908-a04a0360709f.png">
<img width="350" alt="image" src="https://user-images.githubusercontent.com/45623734/122233715-ec24ec80-ced9-11eb-9607-3ff2b05eb315.png">


* Machine Learning Model 2 Training

<img width="416" alt="image" src="https://user-images.githubusercontent.com/45623734/122233841-02cb4380-ceda-11eb-98ca-da397812d577.png">

* Machine Learning Model 2 Evaluation Metrics

<img width="368" alt="image" src="https://user-images.githubusercontent.com/45623734/122233877-08c12480-ceda-11eb-832f-38fc09ab90b0.png">
<img width="368" alt="image" src="https://user-images.githubusercontent.com/45623734/122233895-0c54ab80-ceda-11eb-8f75-445d52ceb817.png">
<img width="368" alt="image" src="https://user-images.githubusercontent.com/45623734/122233915-0f4f9c00-ceda-11eb-912c-c7ae370a9603.png">


### Prediction Result

<img width="349" alt="image" src="https://user-images.githubusercontent.com/45623734/122234056-2d1d0100-ceda-11eb-97eb-d59ece933d1b.png">
<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234065-3017f180-ceda-11eb-86d6-de5b1cdbb725.png">


#### Comparative Analysis on both the Machine Learning Models

* Model 1 Max Train Accuracy = 83%
* Model 2 Max Train Accuracy = 94%

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234148-42922b00-ceda-11eb-95ac-f712bcae73c3.png">

* Model 1 Min Train Loss = 45%
* Model 2 Min Train Loss = 14%

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234184-4d4cc000-ceda-11eb-9c1c-db9a85d2e1dd.png">

* Model 1 Max Validation Accuracy = 84%
* Model 2 Max Validation Accuracy =  92%

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234248-5d649f80-ceda-11eb-87c8-ebc3c4bc8a5a.png">

* Model 1 Min Validation Loss = 6%
* Model 2 Min Validation Loss =  17%

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234270-635a8080-ceda-11eb-89bd-398021f8ac6e.png">

* Model 1 Max Kappa Score = 0.90
* Model 2 Max Kappa Score = 0.78 

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234297-69506180-ceda-11eb-9f9a-3632896dd47b.png">

* Difference in Predicted values by both Models

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234450-808f4f00-ceda-11eb-8e11-72b344d694f5.png">


* Comparison of Predicted Severity Level by both Models

<img width="359" alt="image" src="https://user-images.githubusercontent.com/45623734/122234495-8be27a80-ceda-11eb-99c0-07eeb9caf341.png">


### Conclusion

The motive behind the paper was to develop a system based on ML techniques for monitoring and helping individuals, doctors, health staff, etc. and hence, provide timely help to the people in need of medical assistance. This work will be useful in identifying possible patients who may suffer from Diabetic Retinopathy. This may help in taking preventive measures and hence try to avoid the possibility of Total Blindness caused by this disease. 

In this paper, a novel DenseNet-based deep neural network was presented to predict severity level of diabetic retinopathy in retinal image scan and, in turn, help in early-stage treatments. Machine learning techniques were used to process raw images and provide novel insights towards Diabetic Retinopathy disease. This system extracts the fundal highlights, for example, retinal veins, optic circle, exudates, cottonwool spots, hemorrhages, and microaneurysms  for a low-quality color fundus image. In this work, we have presented two Models based on the DenseNet network with several pre-processing methods to increase the performance of the architecture. Our Model 1 achieved 84% validation accuracy with a kappa score of more than 0.90, whereas our Model 2 achieved 92% validation accuracy with a kappa score of more than 0.78 in severity grading on the challenging Kaggle APTOS 2019 Blindness Detection dataset. The trial results have exhibited the adequacy of our proposed calculation to be sufficient to be utilized in clinical applications.





    
