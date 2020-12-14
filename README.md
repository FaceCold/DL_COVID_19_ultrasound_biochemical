# DL_COVID_19_ultrasound_sound
This package provides an implementation of the assessment and prognosis prediction of COVID-19 patients using lung ultrasound images.

## Prerequisites
* Python 3.5.2
* Pytorch
* OpenCV (for generating CAMs)
* MATLAB R2019b

## data
Ultrasound images are in jpg format. Biochemical indices are in xlsx format.

### input data
* To do the scoring task of ultrasound images, ultrasound images in jpg format are expected to be input to our AI model.
* To do the prognosis prediction of COVID-19 patients, a xlsx file in following format is expected.

| patient ID        | US    |  LYMPH  |  CRP  |  LDH  |  PCT  |  IL-6  |
| -----: | -----:| -----:| -----:| -----:| -----:| -----: |
| 1      |       |       |       |       |       |        |
| 2      |       |       |       |       |       |        |
| 3      |       |       |       |       |       |        |

## Usage
* Use the **runTest()** function in the **Main.py** to obtain the probablities of each score (0, 1, 2, 3) for certain ultrasound images.
* Run **python Main.py** to run the scoring task.
* Run **xxx.m** to do the classification task of mild and severe COVID-19 patients.
* Run **xxy.m** to do the prognosis prediction task of COVID-19 patients.
