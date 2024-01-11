import collections
import itertools
import unittest

import numpy as np
import torch

import viscy.utils.cli_utils as io_utils
from viscy.unet.networks.layers.ConvBlock3D import ConvBlock3D


class TestConvBlock3D(unittest.TestCase):
    """
    Testing class for all configurations of the 3d conv block
    Functionality of core PyTorch and nummpy operations assumed to be
    complete and sound.
    """

    def SetUp(self):
        """
        Set up inputs and block configurations
        """
        # possible inputs and output shapes
        self.pass_inputs = {
            "standard": [torch.ones((1, 1, 5, 256, 256)), (1, 4, 5, 256, 256)],
            "down": [torch.ones((1, 8, 3, 16, 16)), (1, 4, 3, 16, 16)],
            "batch": [torch.ones((8, 1, 3, 16, 16)), (8, 4, 3, 16, 16)],
            "deep": [torch.ones((1, 1, 30, 16, 16)), (1, 4, 30, 16, 16)],
            "small": [torch.ones((1, 1, 4, 4, 4)), (1, 4, 4, 4, 4)],
        }
        self.fail_inputs = {
            "nonsquare": [torch.ones((1, 1, 4, 16, 8)), (1, 1, 4, 16, 8)],
            "wrong_dims": [torch.ones((1, 1, 1)), (1, 1, 1)],
        }
        # possible configurations
        self.configs = {
            "dropout": (False, 0.25),
            "norm": ("batch", "instance"),
            "residual": (True, False),
            "activation": ("relu", "leakyrelu", "selu"),
            "transpose": [False],  # True yields padding error in pytorch 1.10
            "kernel_size": (1, (3, 3, 3), (3, 3, 5)),
            "num_layers": (1, 5),
            "filter_steps": ("linear", "first", "last"),
        }

    def _get_outputs(self, kwargs):
        """
        Template testing class

        :param list kwargs: list of arguments for ConvBlock3D object

        :return numpy.ndarray inputs: inputs to convblock
        :return numpy.ndarray outputs: outputs from convblock, respective
        :return tuple exp_out: expected output
        """
        input_, exp_out_shape = (
            self.pass_inputs["standard"][0],
            self.pass_inputs["standard"][1],
        )

        in_filters = input_.shape[1]
        out_filters = exp_out_shape[1]

        block = ConvBlock3D(in_filters, out_filters, *kwargs)

        try:
            output = block(input_)
            input_, output = input_.detach().numpy(), output.detach().numpy()
            exp_out = output
            return input_, output, exp_out
        except:
            input_.detach().numpy()
            return input_, np.ones((1, 1)), np.zeros((1, 1))

    def _get_output_shapes(self, kwargs, pass_):
        """
        Gets outputs for all inputs of type 'pass_'

        If inputs expected to fail, exp_out_shape will be False

        :param list kwargs: list of arguments for ConvBlock3D object
        :param boolean pass_: whether inputs are expected to pass tests

        :return list inputs: list of inputs to convblock
        :return list outputs: list of outputs from convblock, respective
        :return list exp_out_shapes: list of expected output shapes from
                                    convblock, respective
        """
        inputs, outputs, exp_out_shapes = [], [], []
        test_inputs = self.pass_inputs if pass_ else self.fail_inputs
        for test in test_inputs:
            input_, exp_out_shape = test_inputs[test][0], test_inputs[test][1]

            in_filters = input_.shape[1]
            out_filters = exp_out_shape[1]

            block = ConvBlock3D(in_filters, out_filters, *kwargs)

            try:
                output = block(input_)
                inputs.append(input_)
                outputs.append(output)
                exp_out_shapes.append(exp_out_shape)
            except:
                inputs.append(input_)
                outputs.append(False)
                exp_out_shapes.append(exp_out_shape if pass_ else False)

        return inputs, outputs, exp_out_shapes

    def _get_residual_params(self, kwargs, resid_index):
        """
        Gets parameters of residual and nonresidual blocks

        :param list kwargs: list of arguments for ConvBlock3D object
        :param int resid_index: index of residual parameter in kwargs

        :return nn.module.parameter params: trainable parameters of non-residual block
        :return nn.module.parameter resid_params: trainable parameters of residual block
        """
        input_, exp_out_shape = (
            self.pass_inputs["standard"][0],
            self.pass_inputs["standard"][1],
        )

        in_filters = input_.shape[1]
        out_filters = exp_out_shape[1]

        resid_kwargs, kwargs = list(kwargs), list(kwargs)
        kwargs[resid_index] = False
        resid_kwargs[resid_index] = True

        try:
            block = ConvBlock3D(in_filters, out_filters, *kwargs)
            resid_block = ConvBlock3D(in_filters, out_filters, *resid_kwargs)

            return block.parameters(), resid_block.parameters()
        except:
            return None, None

    def _all_test_configurations(self, test, verbose=True):
        """
        Run specified test on all possible ConvBlock3D input configurations.
        Send failure information to stdout.

        Current tests:
            - input->output for cartesian product of parameters
            - shape matching (upsampling, downsampling)
            - residual (same number of trainable params)
            - kernel shapes (nonsquare doesnt break functionality)

        :param str test: which test to run. Must be within {'passing', 'failing', 'residual'}
        :param bool verbose: Verbosity of str output
        """
        self.SetUp()

        configs_list = [self.configs[key] for key in self.configs]
        configs_list = list(itertools.product(*configs_list))
        failed_tests = collections.defaultdict(lambda: [])

        print("Testing", len(configs_list), "configurations:") if verbose else None

        for i, args in enumerate(configs_list):
            if test == "passing":
                # test passing shapes
                _, outputs, exp_out_shapes = self._get_output_shapes(args, True)
                out_shapes = [
                    ar.detach().numpy().shape if isinstance(ar, torch.Tensor) else ar
                    for ar in outputs
                ]
                try:
                    out_shapes = np.array(out_shapes, dtype=object)
                    exp_out_shapes = np.array(exp_out_shapes, dtype=object)
                    fail_message = (
                        f"'Passing' input tests failed on config {i+1} \n args: {args}"
                    )
                    np.testing.assert_array_equal(
                        out_shapes, exp_out_shapes, fail_message
                    )
                except:
                    failed_tests[i].append(args)
            elif test == "failing":
                # test failing shapes
                _, outputs, exp_out_shapes = self._get_output_shapes(args, False)
                out_shapes = [
                    ar.detach().numpy().shape if isinstance(ar, torch.Tensor) else ar
                    for ar in outputs
                ]
                try:
                    out_shapes = np.array(out_shapes, dtype=object)
                    exp_out_shapes = np.array(exp_out_shapes, dtype=object)
                    fail_message = (
                        f"\t'Failing' tests failed on config {i+1} \n args: {args}"
                    )
                    np.testing.assert_array_equal(
                        out_shapes, exp_out_shapes, fail_message
                    )
                except:
                    failed_tests[i].append(args)
            elif test == "residual":
                # test residual
                resid_index = 2
                if args[resid_index] == False:
                    params, resid_params = self._get_residual_params(args, resid_index)
                    try:
                        fail_message = f"\t Residual params tests failed on config {i+1} \n args: {args}"
                        np.testing.assert_equal(
                            len(list(params)), len(list(resid_params)), fail_message
                        )
                    except:
                        failed_tests[i].append(args)

            io_utils.show_progress_bar(configs_list, i, process="testing", interval=10)
        if verbose:
            print(
                f"Testing complete! {len(configs_list)-len(failed_tests)}/{len(configs_list)} passed."
            )
            if len(failed_tests) > 0:
                print(f"Failed messages:")
                for key in failed_tests:
                    print(f"Config {key}: {failed_tests[key]}")

    # -------------- Tests -----------------#

    def test_residual(self):
        """
        Test residual functionality 3D ConvBlock

        Test that residual blocks do not contain additional parameters
        Runs test with every possible block configuration.
        """
        self._all_test_configurations(test="residual")

    def test_passing(self):
        """
        Test passing input functionality 3D ConvBlock

        Test input-output functionality and expected output shape of all passing input shapes.
        Runs test with every possible block configuration.
        """
        self._all_test_configurations(test="passing")

    def test_failing(self):
        """
        Test failing input handling 3D ConvBlock

        Checks to see if all failing input types are caught by conv block.
        Runs test with every possible block configuration.
        """
        self._all_test_configurations(test="failing")
