## Dependencies 

For a straight-forward use of AttDMM, you can install the required libraries from *requirements_attdmm.txt*:
`pip install -r requirements_attdmm.txt`

## Dataset

We made our experiments on MIMIC-III dataset, which is not publicly availiable. In order to access the dataset, please refer to https://mimic.physionet.org/gettingstarted/access/ .

After downloading MIMIC-III, you can use the following github repo for the pre-processing: https://github.com/USC-Melady/Benchmarking_DL_MIMICIII . 

To be aligned with our AttDMM implementation, you need to have the following files under each fold directory.
1. Time-series features:
	1. time_series_training.npy
	1. time_series_val.npy
	1. time_series_test.npy
1. Static features:
	1. nontime_series_training.npy
	1. nontime_series_val.npy
	1. nontime_series_test.npy
1. Time-series masks:
	1. time_series_training_masking.npy
	1. time_series_val_masking.npy
	1. time_series_test_masking.npy
1. Mortality Labels:
	1. y_mor_training.npy
	1. y_mor_val.npy
	1. y_mor_test.npy

## Example Usage

For training:
`python main.py --cuda --experiments_main_folder experiments --experiment_folder default --log attdmm.log --save_model model --save_opt opt --checkpoint_freq 10 --eval_freq 10 --data_folder /home/Data/fold0`

All the log files and the model checkpoints will be saved under *current_dir/experiments_main_folder/experiment_folder/*

for testing:
`python main.py --cuda --experiments_main_folder experiments --experiment_folder default --log attdmm_eval.log --load_model model_best --load_opt opt_best --eval_mode --data_folder /home/Data/fold0`

Note that *experiments_main_folder* and *experiment_folder* have to be consistent with training so that the correct model is loaded properly.  After testing is done, the prediction outputs can be found as *current_dir/experiments_main_folder/experiment_folder/mortality_predictions_test.csv*

For the full set of arguments, please check main.py .
