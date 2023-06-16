import numpy as np
from torch.utils.tensorboard import SummaryWriter

import viscy.evaluation.evaluation_metrics as inference_metrics


class TorchEvaluator(object):
    """
    Handles all procedures involved with model evaluation.

    Params:
    :param dict torch_config: master config file
    """

    def __init__(self, torch_config, device=None) -> None:
        self.torch_config = torch_config

        self.zarr_dir = self.torch_config["zarr_dir"]
        self.network_config = self.torch_config["model"]
        self.training_config = self.torch_config["training"]
        self.dataset_config = self.torch_config["dataset"]
        self.inference_config = self.torch_config["inference"]
        self.preprocessing_config = self.torch_config["preprocessing"]

        self.inference_metrics = {}
        self.log_writer = SummaryWriter(log_dir=self.save_folder)

    def get_save_location(self):
        """
        Sets save location as specified in config files.
        """
        # TODO implement
        return
        # TODO Change the functionality of saving to put inference in the actual
        # train directory the model comes from. Not a big fan

        # model_dir = os.path.dirname(self.inference_config["model_dir"])
        # save_to_train_save_dir = self.inference_config["save_preds_to_model_dir"]

        # if save_to_train_save_dir:
        #     save_dir = model_dir
        # elif "custom_save_preds_dir" in self.inference_config:
        #     custom_save_dir = self.inference_config["custom_save_preds_dir"]
        #     save_dir = custom_save_dir
        # else:
        #     raise ValueError(
        #         "Must provide custom_save_preds_dir if save_preds_to"
        #         "_model_dir is False."
        #     )

        # now = aux_utils.get_timestamp()
        # self.save_folder = os.path.join(save_dir, f"inference_results_{now}")
        # if not os.path.exists(self.save_folder):
        #     os.makedirs(self.save_folder)

    def _collapse_metrics_dict(self, metrics_dict):
        """
        Collapses metrics dict in the form of
            {metric_name: {index: metric,...}}
        to the form
            {metric_name: np.ndarray[metric1, metrics2,...]}

        :param dict metrics_dict: dict of metrics in the first format

        :return dict collapsed_metrics_dict: dict of metrics in the second format
        """
        collapsed_metrics_dict = {}
        for metric_name in metrics_dict:
            val_dict = metrics_dict[metric_name]
            values = [val_dict[index] for index in val_dict]
            collapsed_metrics_dict[metric_name] = np.array(values)

        return collapsed_metrics_dict

    def _get_metrics(
        self,
        target,
        prediction,
        metrics_list,
        metrics_orientations,
        path="unspecified",
        window=None,
    ):
        """
        Gets metrics for this target_/prediction pair in all the specified orientations
        for all the specified metrics.

        :param np.ndarray target: 5d target array (on cpu)
        :param np.ndarray prediction: 5d prediction array (on cpu)
        :param list metrics_list: list of strings
            indicating the name of a desired metric,
            for options see inference.evaluation_metrics. MetricsEstimator docstring
        :param list metrics_orientations: list of strings
            indicating the orientation to compute,
            for options see inference.evaluation_metrics. MetricsEstimator docstring
        :param tuple window: spatial window of this target/prediction pair
            in the larger arrays they come from.

        :return dict prediction_metrics: dict mapping orientation -> pd.dataframe
            of metrics for that orientation
        """
        metrics_estimator = inference_metrics.MetricsEstimator(metrics_list)
        prediction_metrics = {}

        # transpose target and prediction to be in xyz format
        # NOTE: This expects target and pred to be in the format bczyx!
        target = np.transpose(target, (0, 1, -2, -1, -3))
        prediction = np.transpose(prediction, (0, 1, -2, -1, -3))

        zstart, zend = window[0][0], window[0][0] + window[1][0]  # end = start + length
        pred_name = f"slice_{zstart}-{zend}"

        if "xy" in metrics_orientations:
            metrics_estimator.estimate_xy_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_xy = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_xy().to_dict()
            )
            prediction_metrics["xy"] = metrics_xy

        if "xyz" in metrics_orientations:
            metrics_estimator.estimate_xyz_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_xyz = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_xyz().to_dict()
            )
            prediction_metrics["xyz"] = metrics_xyz

        if "xz" in metrics_orientations:
            metrics_estimator.estimate_xz_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_xz = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_xz().to_dict()
            )
            prediction_metrics["xz"] = metrics_xz

        if "yz" in metrics_orientations:
            metrics_estimator.estimate_yz_metrics(
                target=target,
                prediction=prediction,
                pred_name=pred_name,
            )
            metrics_yz = self._collapse_metrics_dict(
                metrics_estimator.get_metrics_yz().to_dict()
            )
            prediction_metrics["yz"] = metrics_yz

        # format metrics
        tag = path + f"_{window}"
        self.inference_metrics[tag] = prediction_metrics

        return prediction_metrics

    def record_metrics(self, sample_information):
        """
        Handles metric recording in tensorboard.

        Metrics are saved position by position.
        If multiple scalar metric values are stored for a
        particular metric in a particular position,
        they are plotted along the axis they are calculated on.

        :param list sample_information: list of tuples containing information about
            each sample in the form
            (position_group, position_path, normalization_meta, window)
        """
        for info_tuple in sample_information:
            _, position_path, normalization_meta, window = info_tuple
            position = position_path.split("/")[-1]
            sample_metrics = self.inference_metrics[position_path + f"_{window}"]

            for orientation in sample_metrics:
                scalar_dict = sample_metrics[orientation]
                pred_name = scalar_dict.pop("pred_name")[0]

                # generate a unique plot & tag for each orientation
                main_tag = f"{position}/{orientation}_{pred_name}"

                # Need to plot a line if metrics calculated along an axis
                if scalar_dict[list(scalar_dict.keys())[0]].shape[0] == 1:
                    self.writer.add_scalars(
                        main_tag=main_tag,
                        tag_scalar_dict=scalar_dict,
                    )
                else:
                    axis_length = scalar_dict[list(scalar_dict.keys())[0]].shape[0]
                    for i in range(axis_length):
                        scalar_dict_i = {}
                        for key in scalar_dict.keys():
                            scalar_dict_i[key] = scalar_dict[key][i]
                        self.writer.add_scalars(
                            main_tag=main_tag,
                            tag_scalar_dict=scalar_dict_i,
                            global_step=i,
                        )
