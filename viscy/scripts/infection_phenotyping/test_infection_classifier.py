# %%
from viscy.data.hcs import HCSDataModule
import lightning.pytorch as pl
from viscy.scripts.infection_phenotyping.classify_infection import SemanticSegUNet2D
from pytorch_lightning.loggers import TensorBoardLogger
from viscy.transforms import NormalizeSampled

# %% test the model on the test set
test_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2024_02_13_DENV_3infMarked_test.zarr"

data_module = HCSDataModule(
    data_path=test_datapath,
    source_channel=['Sensor','Phase'],
    target_channel=['Inf_mask'],
    split_ratio=0.8,
    z_window_size=1,
    architecture="2D",
    num_workers=0,
    batch_size=1,
    normalizations=[
        NormalizeSampled(
            keys=["Sensor", "Phase"],
            level="fov_statistics",
            subtrahend="median",
            divisor="iqr",
        )
    ],
)

data_module.setup(stage="test")

# %% create trainer and input

logger = TensorBoardLogger(
    "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/",
    name="logs_wPhase",
)

trainer = pl.Trainer(
    logger=logger,
    default_root_dir="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase",
    log_every_n_steps=1,
    devices=1,  # Set the number of GPUs to use. This avoids run-time exception from distributed training when the node has multiple GPUs
)

model = SemanticSegUNet2D(
    in_channels=2,
    out_channels=3,
    ckpt_path="/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/models/sensorInf_phenotyping/logs_wPhase/version_34/checkpoints/epoch=99-step=300.ckpt",
)

trainer.test(model=model, datamodule=data_module)




# # %% script to develop confusion matrix for infected cell classifier

# from iohub.ngff import open_ome_zarr
# import numpy as np
# from skimage.measure import regionprops, label
# import cv2
# import seaborn as sns
# import matplotlib.pyplot as plt    

# # %% load the predicted zarr and the human-in-loop annotations zarr

# pred_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/pred/Exp_2024_02_13_DENV_3infMarked_pred.zarr"
# test_datapath = "/hpc/projects/intracellular_dashboard/viral-sensor/infection_classification/datasets/Exp_2024_02_13_DENV_3infMarked_test.zarr"

# pred_dataset = open_ome_zarr(
#     pred_datapath, 
#     layout="hcs",
#     mode="r+",
# )
# chan_pred = pred_dataset.channel_names

# test_dataset = open_ome_zarr(
#     test_datapath, 
#     layout="hcs",
#     mode="r+",
# )
# chan_test = test_dataset.channel_names

# # %% compute confusion matrix for one image
# for well_id, well_data in pred_dataset.wells():
#     well_name, well_no = well_id.split("/")

#     for pos_name, pos_data in well_data.positions():

#             pred_data = pos_data.data
#             pred_pos_data = pred_data.numpy()
#             T,C,Z,X,Y = pred_pos_data.shape

#             test_data = test_dataset[well_id + "/" + pos_name + "/0"]
#             test_pos_data = test_data.numpy()

#             # compute confusion matrix for each time point and add to total
#             conf_mat = np.zeros((2, 2))
#             for time in range(T):
#                 pred_img = pred_pos_data[time, chan_pred.index("Inf_mask_prediction"), 0, : , :]
#                 test_img = test_pos_data[time, chan_test.index("Inf_mask"), 0, : , :]
                
#                 test_img_rz = cv2.resize(test_img, dsize=(2016,2048), interpolation=cv2.INTER_NEAREST)# pred_img = 
#                 pred_img = np.where(test_img_rz > 0, pred_img, 0)

#                 # find objects in every image
#                 label_img = label(test_img_rz)
#                 regions_label = regionprops(label_img)

#                 # store pixel id for every label in pred and test imgs
#                 for region in regions_label:
#                     if region.area > 0:
#                         row, col = region.centroid
#                         pred_id = pred_img[int(row), int(col)]
#                         test_id = test_img_rz[int(row), int(col)]
#                         if pred_id == 1 and test_id == 1:
#                             conf_mat[1,1] += 1
#                         if pred_id == 1 and test_id == 2:
#                             conf_mat[1,0] += 1
#                         if pred_id == 2 and test_id == 1:
#                             conf_mat[0,1] += 1
#                         if pred_id == 2 and test_id == 2:
#                             conf_mat[0,0] += 1

# # display the confusion matrix
# ax= plt.subplot()
# sns.heatmap(conf_mat, annot=True, fmt='g', ax=ax);  #annot=True to annotate cells, ftm='g' to disable scientific notation

# # labels, title and ticks
# ax.set_xlabel('annotated labels');ax.set_ylabel('predicted labels'); 
# ax.set_title('Confusion Matrix'); 
# ax.xaxis.set_ticklabels(['infected', 'uninfected']); ax.yaxis.set_ticklabels(['infected', 'uninfected']);


# # %%
# %%
