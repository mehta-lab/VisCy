from viscy.light.engine import VSUNet
import torch
import gradio as gr
import numpy as np
from numpy.typing import ArrayLike
from skimage import transform, exposure


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
        self.model

    def normalize_fov(self, input: ArrayLike):
        "Normalizing the fov with zero mean and unit variance"
        mean = np.mean(input)
        std = np.std(input)
        return (input - mean) / std

    def predict(self, inp):
        # Setup the Trainer
        # inp = torch.from_numpy(np.array(inp).astype(np.float32))
        # ensure inp is tensor has to be a (B,C,D,H,W) tensor
        inp = self.normalize_fov(inp)
        inp = torch.from_numpy(np.array(inp).astype(np.float32))
        test_dict = dict(
            index=None,
            source=inp.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.model.device),
        )
        with torch.inference_mode():
            self.model.on_predict_start()
            pred = self.model.predict_step(test_dict, 0, 0).cpu().numpy()
        # Return a 2D image
        nuc_pred = pred[0, 0, 0]
        mem_pred = pred[0, 1, 0]
        # pred = np.clip(pred, 0, 1)
        nuc_pred = exposure.rescale_intensity(nuc_pred, out_range=(0, 1))
        mem_pred = exposure.rescale_intensity(mem_pred, out_range=(0, 1))
        return nuc_pred, mem_pred


# %%
if __name__ == "__main__":
    model_ckpt_path = "./VS_models/VSCyto2D/VSCyto2D/epoch=399-step=23200.ckpt"
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
            "./hek.png",
            "./a549.png",
        ],
    ).launch(share=False)
