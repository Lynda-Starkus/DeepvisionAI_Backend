# DeepvisionAI backend
## Fully functional code



The backend functions with Tensorflow, Pytorch and yolov5

- Fully tested on the UCSD pedestrians dataset 
- Easy execution from anaconda prompt

## Features

- Detection of abnormal activities 
- Tracking all pedestrians with high accuracy
- Generating foreground masks
- Uses optical flow features 
- Uses convolutional saptial auto-encoder 





Examples of the output results

>1- preprocessing mask 

![normal_mask](/out90.jpg)

>2- Abnormal behaviour detection

![abnormal_detected](/frame90.jpg)

>3- Full tracking
<img src="/tracking.png" width="240" height="160">


## Files organization

- run .py : the pilot script that runs in order other scripts
- UCSDped1 .py : contains the labeling and reorganization code for the dataset 
- utils .py : takes in the number of tests to be performed as an argument and generates a folder containing only the selected samples 
- test_script .py : loads the pretrained model results from features/ and processes the test sample
- frames2video .py : a script that compiles a given sequence of images to a video
- track .py : runs yolov5 scripts detect and track different classes of objects MOT (Multi-Object Tracker) the run .py restrains the process to class 0 only 'pedestrians'
- bg .py : generates foreground masks as a preprocessing step for optical_flow .py


## Installation & Usage

It's recommanded to use [Anaconda].

Download the UCSD dataset : [UCSD_Anomaly_Dataset.v1p2](https://drive.google.com/file/d/11lXPcWBe75cHTa4qMiOIlX-NDIAfTHov/view?usp=sharing)

Extract it to main directroy ./UCSD_Anomaly_Dataset.v1p2

Install the packages

```
pip install requirements.txt
```
Run the pilot script :
```
python run.py -test [number of tests to be conducted default = 36] -tracking [runs tracking too default = true]
```



## Performance results

Below are the performance results compared to other state-of-the-art results.

| Method | ROC AUC|
| ------ | ------ |
| Our method | 0.91588 |
| self trained deep ordinal regression | 0.927 |
| full-BVP | 0.836 |
| H-MDT CRF | 0.827 |
| STRT unsupervised  | 0.5945 |
| STRT supervised | 0.7118 |

> Roc curve for 36 tests
<img src="/roc_curve.png" width="600" height="200">

## References for statistics 

[Ref 1] : https://www.researchgate.net/publication/293042967_Anomaly_detection_based_on_spatio-temporal_sparse_representation_and_visual_attention_analysis/figures?lo=1

[Ref 2] : https://www.google.com/url?sa=i&url=https%3A%2F%2Fwww.researchgate.net%2Ffigure%2FROC-curves-of-pixel-level-criterion-on-Ped1_fig1_239943156&psig=AOvVaw2NlY3dVHMzMqlnocK6xTd7&ust=1631800725361000&source=images&cd=vfe&ved=0CAoQjhxqFwoTCKCD6MiRgfMCFQAAAAAdAAAAABAD

## ©️ License

CERIST : Centre de Recherche sur l'Information Scientifique et Technique

ESI : Ecole Nationale Supérieure d'Informatique d'Alger (Ex. INI)

Team :
- Linda Belkessa (Chef d'équipe) https://www.linkedin.com/in/linda-belkessa/
- Sarah Abchiche https://www.linkedin.com/in/sarah-abchiche/
- Salima Mamma https://www.linkedin.com/in/salima-mamma-239002179/
- Sofia Ouanes https://www.linkedin.com/in/sofia-ouanes-18a841182/
- Abdelaziz Takouche https://www.linkedin.com/in/abdelaziz-takouche-9a990b204/
- Massinissa Si Ahmed  https://www.linkedin.com/in/massinissa-si-ahmed-0463a316b/

For more informations please refer to one of the members of the team
**It's totally free for use under license**

