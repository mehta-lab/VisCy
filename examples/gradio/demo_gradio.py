import gradio as gr
import torch
from viscy.light.engine import VSUNet
from huggingface_hub import hf_hub_download
from numpy.typing import ArrayLike
import numpy as np
from skimage import exposure


class VSGradio:
    def __init__(self, model_config, model_ckpt_path):
        self.model_config = model_config
        self.model_ckpt_path = model_ckpt_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.load_model()

    def load_model(self):
        # Load the model checkpoint and move it to the correct device (GPU or CPU)
        self.model = VSUNet.load_from_checkpoint(
            self.model_ckpt_path,
            architecture="UNeXt2_2D",
            model_config=self.model_config,
        )
        self.model.to(self.device)  # Move the model to the correct device (GPU/CPU)
        self.model.eval()

    def normalize_fov(self, input: ArrayLike):
        "Normalizing the fov with zero mean and unit variance"
        mean = np.mean(input)
        std = np.std(input)
        return (input - mean) / std

    def predict(self, inp):
        # Normalize the input and convert to tensor
        inp = self.normalize_fov(inp)
        inp = torch.from_numpy(np.array(inp).astype(np.float32))

        # Prepare the input dictionary and move input to the correct device (GPU or CPU)
        test_dict = dict(
            index=None,
            source=inp.unsqueeze(0).unsqueeze(0).unsqueeze(0).to(self.device),
        )

        # Run model inference
        with torch.inference_mode():
            self.model.on_predict_start()  # Necessary preprocessing for the model
            pred = (
                self.model.predict_step(test_dict, 0, 0).cpu().numpy()
            )  # Move output back to CPU for post-processing

        # Post-process the model output and rescale intensity
        nuc_pred = pred[0, 0, 0]
        mem_pred = pred[0, 1, 0]
        nuc_pred = exposure.rescale_intensity(nuc_pred, out_range=(0, 1))
        mem_pred = exposure.rescale_intensity(mem_pred, out_range=(0, 1))

        return nuc_pred, mem_pred


# Load the custom CSS from the file
def load_css(file_path):
    with open(file_path, "r") as file:
        return file.read()


# %%
if __name__ == "__main__":
    # Download the model checkpoint from Hugging Face
    model_ckpt_path = hf_hub_download(
        repo_id="compmicro-czb/VSCyto2D", filename="epoch=399-step=23200.ckpt"
    )

    # Model configuration
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

    # Initialize the Gradio app using Blocks
    with gr.Blocks(css=load_css("style.css")) as demo:
        # Title and description
        gr.HTML(
            "<div class='title-block'>Image Translation (Virtual Staining) of cellular landmark organelles</div>"
        )
        # Improved description block with better formatting
        gr.HTML(
            """
            <div class='description-block'>
                <p><b>Model:</b> VSCyto2D</p>
                <p>
                    <b>Input:</b> label-free image (e.g., QPI or phase contrast) <br>
                    <b>Output:</b> two virtually stained channels: one for the <b>nucleus</b> and one for the <b>cell membrane</b>.
                </p>
                <p>
                    Check out our preprint: 
                    <a href='https://www.biorxiv.org/content/10.1101/2024.05.31.596901' target='_blank'><i>Liu et al.,Robust virtual staining of landmark organelles</i></a>
                </p>
            </div>
            """
        )

        vsgradio = VSGradio(model_config, model_ckpt_path)

        # Layout for input and output images
        with gr.Row():
            input_image = gr.Image(type="numpy", image_mode="L", label="Upload Image")
            with gr.Column():
                output_nucleus = gr.Image(type="numpy", label="VS Nucleus")
                output_membrane = gr.Image(type="numpy", label="VS Membrane")

        # Button to trigger prediction
        submit_button = gr.Button("Submit")

        # Define what happens when the button is clicked
        submit_button.click(
            vsgradio.predict,
            inputs=input_image,
            outputs=[output_nucleus, output_membrane],
        )

        # Example images and article
        gr.Examples(
            examples=["examples/a549.png", "examples/hek.png"], inputs=input_image
        )

        # Article or footer information
        gr.HTML(
            """
            <div class='article-block'>
            <p> Model trained primarily on HEK293T, BJ5, and A549 cells. For best results, use quantitative phase images (QPI) or Zernike phase contrast.</p>
            <p> For training, inference and evaluation of the model refer to the <a href='https://github.com/mehta-lab/VisCy/tree/main/examples/virtual_staining/dlmbl_exercise' target='_blank'>GitHub repository</a>.</p>
            </div>
            """
        )

    # Launch the Gradio app
    demo.launch()
