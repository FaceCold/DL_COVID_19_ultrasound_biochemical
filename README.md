# Assessment and Prognosis Prediction of COVID-19 Patients Using Lung Ultrasound Images
This package provides an implementation of the assessment and prognosis prediction of COVID-19 patients using lung ultrasound images.

## Prerequisites
* Python 3.5.2
* Pytorch
* OpenCV (for generating CAMs)
* MATLAB R2019b

## data
Ultrasound images are in jpg format. Biochemical indices are in xlsx format. Intermediate files are in txt or mat format.

### Input data
* To do the scoring task of ultrasound images, ultrasound images in jpg format are expected to be input to our AI model. And the name of ultrasound images is expected to be named after *patientID_location_score*, *6_R2_0.jpg* for example. Plus, a single txt file containing file names and labels of all the images is also expected. The txt file should be organized as following.

    1_L1_2.jpg 0 0 1 0  
    1_L2_1.jpg 0 1 0 0  
    2_L1_0.jpg 1 0 0 0  
    2_L2_0.jpg 1 0 0 0  
    ...  
    Where 1 0 0 0 stands for score 0, 0 1 0 0 stands for score 1, 0 0 1 0 stands for score 2, 0 0 0 1 stands for score 3.
* To do the classification task of mild and severe cases, a txt file containing scoring results of each ultrasound image is expected.
* To do the prognosis prediction task of COVID-19 patients, a xlsx file in following format is expected.
  
| patient ID   | US        |  LYMPH      |  CRP       |  LDH      |  PCT       |  IL-6      |
| -----:       | -----:    | -----:      | -----:     | -----:    | -----:     | -----:     |
| 1            | 1.28      |  1.91       |  3.50      |  195      |   0.03     |   3.3      |
| 2            | 0.63      |  1.88       |  3.07      |  285      |            |   1.5      |
| 3            | 0.14      |  2.21       |  2.13      |           |   0.08     |   2.37     |

If the value of certain indices is unknow or missing, the cell is left empty.  
Explanations of each element are as followings.

| Element          | Data type    |  Unit       |  Explanation       |
| -----:           | -----:       | -----:      | :-----             | 
| Patient ID       | intiger      |  none       |  ID of patients      | 
| US               | float        |  none       |  Ultrasound Scores, the average ultrasound scores of each patient obtained from AI model  | 
| LYMPH            | float        |  10^9/L     |  The absolute value of Lymphocyte    | 
| CRP              | float        |  mg/L       |  C-reactive protein    | 
| LDH              | float        |  IU/L       |  Lactate dehydrogenase   | 
| PCT              | float        |  ng/ml      |  Procalcitonin    | 
| IL-6             | float        |  pg/ml      |  Interleukin-6    | 

### Output data
* For the scoring task, AI model outputs the probability of score 0, 1, 2, 3. We choose the score with highest probability as the final score of certain ultrasound images.
* For the classification task of mild and severe cases, AI model outputs a binary value of 0 or 1. 0 stands for mild cases and 1 stands for severe cases.
* For the prognosis prediction task, AI model outputs a binary value of 0 or 1. 0 stands for survival and 1 stands for death.

### File explanation
* **score_database** is the folder containing ultrasound images.
* **dataset** is the folder containing the txt format of test set of ultrasound images.
* **ScoringModel.tar** is the weights of our AI model.
* **cams.py** is the python file to visualize the output of AI model.
* **NetTester.py** is the python file to implement the AI model.
* **DatasetGenerator.py** is the python file to split data into train set and test set.
* **Models.py** is the python file to construct the backbone of AI Model (Resnet18).
* **Main.py** is the python file to run the scoring task.
* **preprocess.m** is the MATLAB file to do some data transform.
* **svmThreshold_CNN.m** is the MATLAB file to do the classification task of mild and severe cases.
* **KNNDecision.m** is the MATLAB file to do the progonosis prediction task of COVID-19 patients.

## Usage
1. Put all the test ultrasound images in the folder **score_database** and copy their file names and labels into the txt file **score_test.txt**.  
2. Run  
    `python Main.py`  
3. Copy the true and predicted scoring results into **gt.txt** and **pred.txt** respectively.
4. Run **preprocess.m** using MATLAB2019b and save the variate ***pred_after*** to **pred.mat**.
5. Run **svmThreshold_CNN.m** using MATLAB2019b to do the the classification task of mild and severe cases.
6. Read the xlsx file of patients and save the 6 biochemical indices along with the ultrasound scores to **DrawScatter.mat**.
7. Run **KNNDecision.m** using MATLAB2019b to do the prognosis prediction task.

## Evaluation
Evaluations are included in **svmThreshold_CNN.m** and **KNNDecision.m**.  
Following indices are used for evaluations.

| Index              |       Explaination       |
| :-----             |        :-----            |
| Accuracy           |         The percentage of correctly predicted results in the total sample           | 
| Precision          |         The proportion of correctly classified positive samples among all samples predicted to be positive            |
| Recall/Sensitivity |         The probability of predicting a correctly classified positive sample among all actually positive samples            |
| F1score            |         The comprehensive measure of model precision and recall            |

## Disclaimer
Our work is only for research purpose and is not approved for clinical use.
