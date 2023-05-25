# Error correcting 2D-3D cascaded network for myocardial infarct segmentation



This repository is created for learning non-linear regularizing filters for inverting the Radon transform. For more detail about non-linear regularizing filters see:

```
Ebner, A., & Haltmeier, M. (2023). When noise and data collide: Using non-linear Filters to save the day in inverse problems. Best Journal you can imagine, 62, 66-83.
```


# Instalation

1. Clone the git repository. 
```
git clone https://git.uibk.ac.at/c7021123/error-correcting-2d-3d-cascaded-network-for-myocardial-infarct-segmentation.git
``` 

2. Intall and activate the virtual environment.
```
cd learned-filters
conda env create -f env_lge.yml
conda activate env_lge
``` 

# Usage

## Preprocessing
1. Download the [EMIDEC Dataset](https://emidec.com/dataset#download). Make shure to download both train and test datasets and save them in a folder called `DATA/` 
``` 
DATA/
├── emidec-dataset-1.0.1 
├── emidec-segmentation-testset-1.0.0
```
Note that for training the testset folder is optional and does not have to be present.

2. Prepare the downloaded dataset for training. For this run the following command in your console
```
python3 preprocessing.py 
``` 


## Training

### 2D U-Net

To train the two dimensional U-Nets run the command
```
python3 2d-net.py --fold FOLD 
``` 
- `--fold` specifies on which on which od the five folds the network should be trained 

### 2D-3D cascade

To train the two dimensional U-Nets run the command
```
python3 2d-net.py --fold FOLD 
``` 
- `--fold` specifies on which on which od the five folds the network should be trained 

## Testing

To test the final framework on the testset of the EMICED callenge run 
```
python3 inference.py 

``` 
Predictions and plots of the results will be saved in `RESULTS_FOLDER/testset`.


## Authors and acknowledgment
Matthias Schwab<sup>1,2</sup>, Andrea Ebner<sup>1</sup>

<sup>1</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria

<sup>2</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria



