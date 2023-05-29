import argparse
import numpy as np
import os
import onnxruntime as ort 
import onnx
import pathlib
import sys
import torch.onnx as torch_onnx
import torch

sys.path.insert(0, "/home/christian.foley/virtual_staining/workspaces/microDL")
import micro_dl.inference.inference as inference

def parse_args():
    """
    Parse command line arguments

    In python namespaces are implemented as dictionaries
    :return: namespace containing the arguments passed.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        help="path to yaml configuration file",
    )
    parser.add_argument(
        "--stack_depth",
        required=True,
        help="Stack depth of model. If model is 2D, use 1."
    )
    parser.add_argument(
        "--export_path", 
        required=True, 
        help="Path to store exported model"
    )
    parser.add_argument(
        "--test_input",
        required=False,
        default=None,
        help="Path to .npy test input for additional model validation after export."
    )
    args = parser.parse_args()
    return args


def validate_export(model_path) -> None:
    """
    Run ONNX validation on exported model. Assures export success.

    :param str model_path: path to exported onnx model
    """
    print("Validating model...", end='')
    onnx_model = onnx.load(model_path)
    try:
        onnx.checker.check_model(onnx_model)
        print("Passed!")
    except Exception as e:
        print("Failed:")
        print("\t", e)
        sys.exit()


def remove_initializer_from_input(model_path):
    """
    De-initializes model at model_path, and overwrites with de-initialized version

    :param str model_path: path to model to de-initialize inputs
    """
    model = onnx.load(model_path)
    if model.ir_version < 4:
        print(
            "Model with ir_version below 4 requires to include initilizer in graph input"
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    
    onnx.save(model, model_path)


def export_model(model_dir, model_name, stack_depth, export_path) -> None:
    """
    Export a model to onnx. Due to restrictions in the pytorch-onnx opset conversion,
    opset for 2.5D and 2D unets are limited to version 10 without dropout.

    :param str model_dir: path to model directory
    :param str model_name: name of model in directory
    :patah str export_path: intended path for exported model
    """
    print("Initializing model in pytorch...")
    torch_predictor = inference.TorchPredictor(
        config={"model_dir": model_dir, "model_name": model_name},
        device="cpu",
        single_prediction=True,
    )
    torch_predictor.load_model()
    model = torch_predictor.model
    model.eval()
    
    if stack_depth == 1:
        sample_input = np.random.rand(1, 1, 512, 512)
    else:
        sample_input = np.random.rand(1, 1, stack_depth, 512, 512)
    input_tensor = torch.tensor(sample_input.astype(np.float32), requires_grad=True)
    print("Done!")
    
    # Export the model
    print("Exporting model to onnx...", end="")
    torch_onnx.export(
        model,
        input_tensor, 
        export_path,
        export_params=True,
        opset_version=10,  # the ONNX version to export the model to
        do_constant_folding=True,  # whether to execute constant folding for optimization
        operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN_FALLBACK,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 1: "channels", 3: "num_rows", 4: "num_cols"},
            "output": {0: "batch_size", 1: "channels", 3: "num_rows", 4: "num_cols"},
        },
    )
    remove_initializer_from_input(export_path)
    validate_export(export_path)
    print("Done!")


def infer(model_path, data_path, output_path) -> None:
    """
    Run an inference session with an onnx model. Data will be read into a numpy array
    and stored as a numpy array. 

    :param str model_path: path to onnx model for inference
    :param str data_path: path to data for inference
    :param str output_path: path to save model output to
    """
    
    validate_export(model_path)
    data = np.load(data_path)
    
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1

    ort_sess = ort.InferenceSession(model_path)
    outputs = ort_sess.run(None, {"input": data})
    
    np.save(output_path, outputs)


def main(args):
    model_dir = pathlib.Path(args.model_path).parent.absolute().parent.absolute()
    model_name = os.path.basename(args.model_path)
    
    export_model(model_dir, model_name, args.stack_depth, args.export_path)
    
    # if specified, run test with some test input numpy array
    if args.test_input is not None:
        print("Running inference test with ONNX model on CPU...")
        test_out_dir = pathlib.Path(args.test_input).parent.absolute()
        test_out_name = "test_pred_" + os.path.basename(args.test_input)
        test_out_path = os.path.join(test_out_dir, test_out_name)
        infer(args.export_path, args.test_input, test_out_path)
        print("Done!")

if __name__ == "__main__":
    args = parse_args()
    main(args)        

