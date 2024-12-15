# Predictive Modeling and Risk Analysis in Parkinson's Disease 

![(https://en.wikipedia.org/wiki/Parkinson%27s_disease)](https://www.ox.ac.uk/sites/files/oxford/field/field_image_main/human%20brain.jpg) 

This repository contains the final project of the Fundamentals of Data Science course. Here you can find our analyses performed on a synthetic dataset of patients with and without Parkinson's disease, you can find the dataset we used [here](https://www.kaggle.com/datasets/rabieelkharoua/parkinsons-disease-dataset-analysis/data).

---
### **Objective of the Analysis**

Our goal is to develop an advanced classification system for early diagnosis of Parkinson's disease, using machine learning and artificial intelligence techniques. By analyzing a synthetic dataset containing clinical, demographic and lifestyle profiles, the research aims to identify the most accurate and reliable predictive model.

---
## Repository Contents:

### Parkinsons Modeling and Functions

This folder contains a notebook file with the progress and comments of the performed tasks and several `.py` files with the architecture of the neural networks used for binary classification. Specifically, it includes:

### Files:

- [**main.ipynb**](https://github.com/CamillaMilli/hw-fds-/blob/main/Parkinsons_Modeling_and_Functions/main.ipynb): A Jupyter Notebook containing all the analyses and comparisons of the different models used;

- [**Models**](https://github.com/CamillaMilli/hw-fds-/tree/main/Parkinsons_Modeling_and_Functions/models): This folder contain the codes of two Neural Network. In details:
  
  - [**BinaryClassifierNet.py**](https://github.com/CamillaMilli/hw-fds-/blob/main/Parkinsons_Modeling_and_Functions/models/BinaryClassifierNet.py): A 3-layer neural network with batch normalization, which uses a sigmoid activation function in the output layer to produce a probability vector for each class;

  - [**ImproveClassifierNe.py**](https://github.com/CamillaMilli/hw-fds-/blob/main/Parkinsons_Modeling_and_Functions/models/ImproveClassifierNet.py): A deeper version of the BinaryClassifierNet with 5 layers and featuring two Dropout layers to deactivate some neurons during the training phase to prevent overfitting of the network.

--- 

### **Supervised Models and Neural Networks for Binary Classification**

In this analysis, we employed various supervised machine learning models and two neural networks for binary classification, aiming to predict the target variable *Diagnosis* (0 = No Parkinson's, 1 = Parkinson's Disease). We compared the results of these models using several performance metrics, which include:

#### **Performance Metrics**

1. **Accuracy**
   
   This metric measures the overall correctness of the model. It is the ratio of correctly predicted observations to the total observations.
  
   $\text{Accuracy}$ = $\frac{\text{True Positives} + \text{True Negatives}}{\text{Total Population}}$
   

3. **Recall (Sensitivity)**
   
   Recall, also known as sensitivity or true positive rate, indicates how well the model identifies positive instances (Parkinson’s Disease). It is the ratio of correctly predicted positive observations to all actual positives.
   
   $\text{Recall}$ = $\frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}$

4. **Precision (Positive Predictive Value)**
   
   This metric measures the accuracy of positive predictions. It is the ratio of correctly predicted positive observations to all predicted positives.

   $\text{Precision}$ = $\frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}$
   

5. **F1 Score**
   
   The F1 Score is the harmonic mean of Precision and Recall, providing a balance between them. It is particularly useful when there is an imbalance between classes.

   F1 $\text{ Score}$ =   2 $\times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$
  

6. **ROC AUC (Receiver Operating Characteristic - Area Under Curve)**
   
   The ROC AUC score measures the ability of the model to distinguish between classes. A higher AUC value indicates a better model. The ROC curve plots the True Positive Rate against the False Positive Rate.
  
   $\text{AUC-ROC}$ = $\int_0^1 \text{True Positive Rate} \, d(\text{False Positive Rate})$

By evaluating these metrics, we were able to assess the strengths and weaknesses of each model and select the best-performing ones for Parkinson’s Disease diagnosis prediction.


