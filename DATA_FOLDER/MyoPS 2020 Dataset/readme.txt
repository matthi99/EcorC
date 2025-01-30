
# all released gold standard segmentation of test data are encrypted. please use the "zxhCardMyoPSEvaluate" from zxhproj to evaluate the Dice score of the pathologies.
# e.g. for test_201, the obtained myops is myseg_201_result.nii.gz,
zxhCardMyoPSEvaluate.exe -evaps myops_test_201_gdencrypt.nii.gz myseg_201_result.nii.gz 1

success: open image myops_test_201_gdencrypt.nii.gz !
success: open image myseg_201_result.nii.gz !
dice_scar        dice_edemascar
0.610000        0.655000

BTW: zxhCardMyoPSEvaluate.exe tool include two versions, i.e., one for windows and one for mac.

For MacOSX, the command is similar,
zxhCardMyoPSEvaluate -evaps myops_test_201_gdencrypt.nii.gz myseg_201_result.nii.gz 1
Note that before using this, one needs to set zxhCardMyoPSEvaluate to be an executable file using chmod.

## How to cite

Please cite the following papers when you use the data for publications:
[1] Xiahai Zhuang: Multivariate mixture model for myocardial segmentation combining multi-source images. IEEE Transactions on Pattern Analysis and Machine Intelligence (T PAMI), vol. 41, no. 12, 2933-2946, Dec 2019.
[2] Xiahai Zhuang: Multivariate mixture model for cardiac segmentation from multi-sequence MRI. International Conference on Medical Image Computing and Computer-Assisted Intervention, pp.581-588, 2016.