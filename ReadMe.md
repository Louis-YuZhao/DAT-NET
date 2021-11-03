# DAT-NET
Code for paper titled "Decode the dopamine transporter imaging for differential parkinsonism diagnosis using deep learning" 

by Yu Zhao, University of Bern and Technical University of Munich 

last modified 21.10.2020

Will continue to be updated. 

# Requirement:
  > Python 2.7.3  
  > tensorflow 1.9.0  
  > Keras 2.2.2  
  > keras-contrib 2.0.8  
  > pandas 0.24.2  
  > scikit-image 0.14.0  
  > scikit-learn 0.19.2  
  > SimpleITK 1.1.0  


# Guideline for utilizing:

(1) Prepare the dopamine transporter imaging PET dataset and store the data to folder: "../DATASET"

(2) Modify the config.py document to assign the GPU device, original PET image size, and involved PET image labels (IPD，MSA，PSP).

(3) Data preparing

    python Dataprepare.py

(4) trainging the model:
    
    python train.py

(5) Evaluated the performance of the trained model in cross-validation:

    python evaluate.py

(6) Evaluated the performance of the trained model in the blind test:

    python Evaluation_On_TestDataset.py