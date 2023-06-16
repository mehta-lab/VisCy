import collections
import itertools
import unittest

import numpy as np
import torch

import viscy.utils.cli_utils as io_utils
from viscy.unet.networks.Unet2D import Unet2d


class TestUnet2d(unittest.TestCase):
    """
    Testing class for all configurations of the 2D Unet implementaion
    Functionality of core PyTorch and nummpy operations assumed to be
    complete and sound.
    """

    def SetUp(self):
        """
        Set up inputs and block configurations
        """
        # possible inputs and output shapes
        self.pass_inputs = {
            "standard": [torch.ones((1, 1, 256, 256)), (1, 1, 256, 256)],
            "batch": [torch.ones((3, 1, 256, 256)), (3, 1, 256, 256)],
            "multichannel": [torch.ones((1, 1, 256, 256)), (1, 2, 256, 256)],
            "multichannel_flat": [torch.ones((1, 2, 256, 256)), (1, 2, 256, 256)],
        }
        self.fail_inputs = {
            "nonsquare": [torch.ones((1, 1, 128, 256)), (1, 1, 128, 256)],
            "nonsquare_arbitrary": [torch.ones((1, 1, 128, 316)), (1, 1, 128, 316)],
            "wrong_dims": [torch.ones((1, 1, 1)), (1, 1, 1)],
        }
        # possible configurations
        self.configs = {
            "xy_kernel_size": ((1, 1), (3, 5), (3, 3)),
            "residual": (True, False),
            "dropout": (False, 0.25),
            "num_blocks": (1, 2, 4),
            "num_block_layers": (1, 3),  # True yields padding error in pytorch 1.10
            "num_filters": ([],),
            "task": ("reg", "seg"),
        }

    def _get_outputs(self, kwargs):
        """
        Template testing class

        :param list kwargs: list of arguments for Unet object

        :return numpy.ndarray inputs: inputs to Unet
        :return numpy.ndarray outputs: outputs from Unet, respective
        :return tuple exp_out: expected output
        """
        input_, exp_out_shape = (
            self.pass_inputs["standard"][0],
            self.pass_inputs["standard"][1],
        )

        in_channels = input_.shape[1]
        out_channels = exp_out_shape[1]

        network = Unet2d(in_channels, out_channels, *kwargs)

        try:
            output = network(input_)
            input_, output = input_.detach().numpy(), output.detach().numpy()
            exp_out = output
            return input_, output, exp_out
        except Exception as e:
            self.excep = e
            input_.detach().numpy()
            return input_, np.ones((1, 1)), np.zeros((1, 1))

    def _get_output_shapes(self, kwargs, pass_):
        """
        Gets outputs for all inputs of type 'pass_'

        If inputs expected to fail, exp_out_shape will be False

        :param list kwargs: list of arguments for Unet2d object
        :param boolean pass_: whether inputs are expected to pass tests

        :return list inputs: list of inputs to Unet
        :return list outputs: list of outputs from Unet, respective
        :return list exp_out_shapes: list of expected output shapes from
                                    Unet, respective
        """
        inputs, outputs, exp_out_shapes = [], [], []
        test_inputs = self.pass_inputs if pass_ else self.fail_inputs
        for test in test_inputs:
            input_, exp_out_shape = test_inputs[test][0], test_inputs[test][1]

            in_channels = input_.shape[1]
            out_channels = exp_out_shape[1]

            network = Unet2d(in_channels, out_channels, *kwargs)

            try:
                output = network(input_)
                inputs.append(input_)
                outputs.append(output)
                exp_out_shapes.append(exp_out_shape)
            except Exception as e:
                self.excep = e
                inputs.append(input_)
                outputs.append(False)
                exp_out_shapes.append(exp_out_shape if pass_ else False)

        return inputs, outputs, exp_out_shapes

    def _get_residual_params(self, kwargs, resid_index):
        """
        Gets parameters of residual and nonresidual networks

        :param list kwargs: list of arguments for Unet2d object
        :param int resid_index: index of residual parameter in kwargs

        :return nn.module.parameter params: trainable parameters of non-residual block
        :return nn.module.parameter resid_params: trainable parameters of residual block
        """
        input_, exp_out_shape = (
            self.pass_inputs["standard"][0],
            self.pass_inputs["standard"][1],
        )

        in_channels = input_.shape[1]
        out_channels = exp_out_shape[1]

        resid_kwargs, kwargs = list(kwargs), list(kwargs)
        kwargs[resid_index] = False
        resid_kwargs[resid_index] = True

        try:
            network = Unet2d(in_channels, out_channels, *kwargs)
            resid_network = Unet2d(in_channels, out_channels, *resid_kwargs)

            return network.parameters(), resid_network.parameters()
        except Exception as e:
            self.excep = e
            return None, None

    def _all_test_configurations(self, test, verbose=True):
        """
        Run specified test on all possible 2d Unet input configurations.
        Send failure information to stdout.

        Current tests:
            - Initialization and input->output for cartesian product of parameters
            - shape matching (single-channel, multi-channel)
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
                    failed_tests[i].append(self.excep)
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
                    failed_tests[i].append(self.excep)
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
                        failed_tests[i].append(self.excep)

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
        Test residual functionality 2D Unet

        Test that residual blocks do not contain additional parameters
        Runs test with every possible block configuration.
        """
        self._all_test_configurations(test="residual")

    def test_passing(self):
        """
        Test passing input functionality 2D Unet

        Test input-output functionality and expected output shape of all passing input shapes.
        Runs test with every possible block configuration.
        """
        self._all_test_configurations(test="passing")

    def test_failing(self):
        """
        Test failing input handling 2D Unet

        Checks to see if all failing input types are caught by conv block.
        Runs test with every possible block configuration.
        """
        self._all_test_configurations(test="failing")
