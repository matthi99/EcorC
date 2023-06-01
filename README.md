# Error correcting 2D-3D cascaded network for myocardial infarct segmentation



This repository is created for the publication:

```
Schwab, M., Pamminger, M., Kremser, C., Obmann, D., Haltmeier, M., Mayr, A. (2023). Error correcting 2D-3D cascaded network for myocardial infarct scar segmentation on late gadolinium enhancement cardiac magnetic resonance images. Best Journal you can imagine, 62, 66-83.
```


# Instalation

1. Clone the git repository. 
```
git clone https://git.uibk.ac.at/c7021123/error-correcting-2d-3d-cascaded-network-for-myocardial-infarct-segmentation.git
``` 

2. Intall and activate the virtual environment.
```
cd error-correcting-2d-3d-cascaded-network-for-myocardial-infarct-segmentation
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
python 2d-net.py --fold XXX 
``` 
- `--fold` specifies on which on which od the five folds (0,1,2,3,4) the network should be trained 

### 2D-3D cascade

To train the Error correcting 2D-3D cascaded framewor run the command
```
python 3d-cascade.py --fold XXX
``` 
Note that to be able to train the cascade the 2D U-Net had to be trained beforehand. 

## Testing

To test the final framework on the testset of the EMICED callenge run 
```
python inference.py 

``` 
Predictions and plots of the results will be saved in `RESULTS_FOLDER/testset`.


## Authors and acknowledgment
Matthias Schwab<sup>1</sup>, Mathias Pamminger<sup>1</sup>, Christian Kremser <sup>1</sup>, Daniel Obmann <sup>2</sup>, Markus Haltmeier <sup>2</sup>, Agnes Mayr <sup>1</sup>

<sup>1</sup> University Hospital for Radiology, Medical University Innsbruck, Anichstraße 35, 6020 Innsbruck, Austria 

<sup>2</sup> Department of Mathematics, University of Innsbruck, Technikerstrasse 13, 6020 Innsbruck, Austria



