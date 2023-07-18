# %% This script allows us to export visualizations from tensorboard logs
# written by lightning training CLI.

import matplotlib.pyplot as plt
import torch

# Get the path to the tensorboard event file
event_file_path = (
    "/hpc/projects/CompMicro/projects/virtual_staining/models/"
    "phase2nuclei018/lightning_logs/20230514-003340/"
    "events.out.tfevents.1684049623.gpu-c-1.2407720.0"
)

# Read the event file
with open(event_file_path, "rb") as f:
    event_data = torch.utils.tensorboard.load_event_file(f)

# Seeing this error:
# AttributeError: module 'torch.utils' has no attribute 'tensorboard'.
# tensorboard 2.12.0 is installed in the environment. Not sure why this error.

# Get the scalars
loss_train = []
loss_val = []
loss_train_step = []
for event in event_data:
    for value in event.summary.value:
        if value.tag == "loss/train_epoch":
            loss_train.append(value.simple_value)
        elif value.tag == "loss/val":
            loss_val.append(value.simple_value)
        elif value.tag == "loss/train_step":
            loss_train_step.append(value.simple_value)

# %% Plot the scalars
plt.plot(loss_train, label="train")
plt.plot(loss_val, label="val")
plt.plot(loss_train_step, label="train_step")
plt.legend()
plt.show()

# Export the images with tag 'val_samples' as mp4
images = []
for event in event_data:
    for image in event.summary.value:
        if image.tag == "val_samples":
            images.append(image.image.encoded_image_string)

# Create a video writer
writer = torch.utils.tensorboard.writer.FFmpegWriter("./val_samples.mp4")

# Write the images to the video writer
for image in images:
    writer.write(image)

# Close the video writer
writer.close()
