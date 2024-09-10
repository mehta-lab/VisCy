# %%
from PIL import Image
from torchvision import transforms

from viscy.data.hcs import HCSDataModule
from viscy.light.engine import FcmaeUNet
from viscy.light.predict_writer import HCSPredictionWriter
from viscy.light.trainer import VSTrainer
from viscy.light.engine import VSUNet
from viscy.transforms import NormalizeSampled
import torch
import gradio as gr
import numpy as np
from numpy.typing import ArrayLike
from skimage import io, transform, exposure


class VSGradio:
    def __init__(self, model_config, model_ckpt_path):
        self.model_config = model_config
        self.model_ckpt_path = model_ckpt_path
        self.model = None
        self.load_model()

    def load_model(self):
        # Load the model checkpoint
        self.model = VSUNet.load_from_checkpoint(
            self.model_ckpt_path,
            architecture="UNeXt2_2D",
            model_config=self.model_config,
            accelerator="gpu",
        )
        self.model.eval()

    def normalize_fov(self, input: ArrayLike):
        "Normalizing the fov with zero mean and unit variance"
        mean = np.mean(input)
        std = np.std(input)
        return (input - mean) / std

    def resize_to_divisible_by_32(self, img):
        # Load the image using skimage

        # Get the current height and width of the image
        height, width = img.shape[-2:]

        # Calculate the new width and height to be divisible by 32
        new_width = (width // 32) * 32
        new_height = (height // 32) * 32

        # Resize the image if needed
        if width != new_width or height != new_height:
            img_resized = transform.resize(
                img, (new_height, new_width), anti_aliasing=True
            )
        else:
            img_resized = img

        return img_resized

    def predict(self, inp):
        # normalize the input
        inp = self.resize_to_divisible_by_32(inp)
        inp = self.normalize_fov(inp)
        inp = torch.from_numpy(np.array(inp).astype(np.float32))
        # ensure inp is tensor has to be a (B,C,D,H,W) tensor
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
            inp = inp.unsqueeze(0)
            inp = inp.unsqueeze(0)

        elif len(inp.shape) == 3:
            inp = inp.unsqueeze(0)
            inp = inp.unsqueeze(0)

        elif len(inp.shape) == 4:
            inp = inp.unsqueeze(0)

        elif len(inp.shape) == 5:
            pass

        with torch.inference_mode():
            inp = inp.to(self.model.device)
            pred = self.model(inp).cpu().numpy()

        # Return a 2D image
        nuc_pred = pred[0, 0, 0]
        mem_pred = pred[0, 1, 0]
        # pred = np.clip(pred, 0, 1)
        nuc_pred = exposure.rescale_intensity(nuc_pred, out_range=(0, 1))
        mem_pred = exposure.rescale_intensity(mem_pred, out_range=(0, 1))
        return nuc_pred, mem_pred


# %%
if __name__ == "__main__":
    input_data_path = ""
    model_ckpt_path = "/hpc/projects/comp.micro/virtual_staining/datasets/public/VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"

    model_config = {
        "in_channels": 1,
        "out_channels": 2,
        "encoder_blocks": [3, 3, 9, 3],
        "dims": [96, 192, 384, 768],
        "decoder_conv_blocks": 2,
        "stem_kernel_size": [1, 2, 2],
        "in_stack_depth": 1,
        "pretraining": False,
    }

    vsgradio = VSGradio(model_config, model_ckpt_path)

    gr.Interface(
        fn=vsgradio.predict,
        inputs=gr.Image(type="numpy", image_mode="L", format="png"),
        outputs=[
            gr.Image(type="numpy", format="png"),
            gr.Image(type="numpy", format="png"),
        ],
        examples=[
            "/home/eduardo.hirata/repos/viscy/examples/gradio/hek.png",
            "/home/eduardo.hirata/repos/viscy/examples/gradio/dog.jpeg",
        ],
    ).launch()
