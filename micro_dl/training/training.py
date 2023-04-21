import datetime
import gunpowder as gp
import numpy as np
import os
import matplotlib.pyplot as plt
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader

if __name__ == "__main__":
    # This should exclusively be set in main file. Cannot set context twice
    torch.multiprocessing.set_start_method("spawn")

import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.input.dataset as ds
import micro_dl.utils.cli_utils as io_utils
import micro_dl.utils.aux_utils as aux_utils


class TorchTrainer:
    """
    TorchTrainer object which handles all the procedures involved with training a pytorch model.
    The trainer uses a the model.py and dataset.py utility modules to instantiate and load a model and
    training data using a gunpowder backend.

    Functionality of the class can be achieved without specifying full torch_config. However, full
    functionality requires full configuration file.
    """

    def __init__(self, torch_config):
        self.torch_config = torch_config

        self.zarr_dir = self.torch_config["zarr_dir"]
        self.network_config = self.torch_config["model"]
        self.training_config = self.torch_config["training"]
        self.dataset_config = self.torch_config["dataset"]

        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None

        self.model = None

        # init token specific parameters - optimizer, loss, device
        # optimizer
        assert self.training_config["optimizer"] in {
            "adam",
            "sgd",
        }, "optimizer must be 'adam' or 'sgd'"
        if self.training_config["optimizer"] == "adam":
            self.optimizer = optim.Adam
        else:
            self.optimizer = optim.SGD
        # lr scheduler
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau
        # loss
        assert self.training_config["loss"] in {"mse", "l1", "cossim", "cel"}, (
            f"loss not supported. " "Try one of 'mse', 'mae', 'cossim', 'l1'"
        )
        if self.training_config["loss"] == "mse":
            self.criterion = nn.MSELoss()
        elif self.training_config["loss"] in {"mae", "l1"}:
            self.criterion = nn.L1Loss()
        elif self.training_config["loss"] == "cossim":
            self.criterion = nn.CosineSimilarity()
        else:
            raise AttributeError(
                f"Loss {self.training_config['loss']} not supported."
                "Try one of 'mse', 'mae' or 'l1', 'cossim'"
            )
        # device
        assert self.training_config["device"] in {
            "cpu",
            "gpu",
            *range(torch.cuda.device_count()),
        }, f"device must be cpu or gpu or within {range(torch.cuda.device_count())}"
        if isinstance(self.training_config["device"], int):
            self.device = torch.device(f"cuda:{self.training_config['device']}")
        elif self.training_config["device"] == "gpu":
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        # plotting
        self.plot = False

    def load_model(self, init_dir=False) -> None:
        """
        Initializes a model according to the network configuration dictionary used to train it, and,
        if provided, loads the parameters saved in init_dir into the model's state dict.

        :param str init_dir: directory containing model weights and biases
        """
        assert (
            self.network_config != None
        ), "Network configuration must be initiated to load model"

        debug_mode = False
        if "debug_mode" in self.network_config:
            debug_mode = self.network_config["debug_mode"]

        model = model_utils.model_init(
            self.network_config,
            device=self.training_config["device"],
            debug_mode=debug_mode,
        )

        if init_dir:
            model_dir = self.network_config["model_dir"]
            readout = model.load_state_dict(torch.load(model_dir))
            print("Initiating from pre-trained model: ", readout)
        self.model = model

        self.model.to(self.device)

    def generate_dataloaders(self, train_key=None, test_key=None, val_key=None) -> None:
        """
        Helper that generates train, test, validation torch dataloaders for loading samples
        into network for training and testing.

        Dataloaders are set to class variables. Dataloaders correspond to one multi-zarr
        dataset each. Each dataset's access key will determine the data array (by type)
        it calls at the well-level.

        If keys unspecified, defaults to the first available data array at each well

        :param int or gp.ArrayKey train_key: key or index of key to data array for training
                                            in training dataset
        :param int or gp.ArrayKey test_key: key or index of key to data array for testing
                                            in testing dataset
        :param int or gp.ArrayKey val_key: key or index of key to data array for validation
                                            in validation dataset
        """
        assert self.torch_config != None, (
            "torch_config must be specified in object" "initiation "
        )
        # init directory for model and metadata storage
        self.get_save_location()

        # init datasets
        workers = 0
        if "num_workers" in self.training_config:
            workers = self.training_config["num_workers"]

        self.data_split = self.get_data_split()
        torch_data_container = ds.TorchDatasetContainer(
            zarr_dir=self.zarr_dir,
            train_config=self.training_config,
            network_config=self.network_config,
            dataset_config=self.dataset_config,
            device=self.device,
            workers=workers,
            use_recorded_split=self.dataset_config["use_recorded_data_split"],
            data_split=self.data_split,
        )
        self.record_data_split(torch_data_container.data_split)

        train_dataset = torch_data_container["train"]
        test_dataset = torch_data_container["test"]
        val_dataset = torch_data_container["val"]

        # initalize dataset keys
        train_key = 0 if train_key == None else train_key
        test_key = 0 if test_key == None else test_key
        val_key = 0 if val_key == None else val_key
        train_dataset.use_key(train_key)
        test_dataset.use_key(test_key)
        val_dataset.use_key(val_key)

        # init dataloaders
        self.train_dataloader = DataLoader(dataset=train_dataset, shuffle=True)
        self.test_dataloader = DataLoader(dataset=test_dataset, shuffle=True)
        self.val_dataloader = DataLoader(dataset=val_dataset, shuffle=True)

    def get_save_location(self):
        """
        Initates save folder if not already initated.
        Directory is named depending on time of training. All training/testing information is
        saved to this directory.
        """
        now = aux_utils.get_timestamp()

        self.save_folder = os.path.join(
            self.training_config["save_dir"], f"training_model_{now}"
        )
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)

    def get_data_split(self):
        """
        Extracts the data split from the model directory referenced in inference. Data
        splits are stored in a .yml local to the model by timestamp. This method will prioritize
        the most recent data split saved. If the data split is not provided, returns false.

        Usage of data split files be overriden by manually inputting the data split positions
        into the inference config section in the config file.

        :return dict data_split: dictionary of data split containing integer list of positions
                            OR decimal fractions indicating split under {'train', 'test', 'val'}
                            keys

        """
        if "custom_data_split" in self.training_config:
            print("Using custom data split found in training config.")
            return self.training_config["custom_data_split"]

        data_split_file = os.path.join(self.save_folder, "data_splits.yml")

        if os.path.exists(data_split_file):
            print("Using saved data split found in model directory.")
            data_splits = aux_utils.read_config(data_split_file)
            timestamps = list(data_splits.keys())
            timestamps.sort(reverse=True)
            most_recent_split = timestamps[0]

            data_split = data_splits[most_recent_split]
            return data_split
        else:
            return self.dataset_config["split_ratio"]

    def record_data_split(self, data_split):
        """
        Records the given data split ('train', 'test', 'val') positions in the given training
        model directory as a yaml file. If data split already recorded for this model, will
        overwrite. The intention here is to record all models separately, and their data splits
        with them.

        :param dict data_split: dictionary of data split containing integer list of positions
                                    under {'train', 'test', 'val'} keys.
        """
        data_split_file = os.path.join(self.save_folder, "data_splits.yml")

        timestamp = aux_utils.get_timestamp()
        if os.path.exists(data_split_file):
            print("Previous split(s) found. Recording current split.")
            data_splits = aux_utils.read_config(data_split_file)
            data_splits[timestamp] = data_split
        else:
            print("No previous split found, storing split in new file")
            data_splits = {timestamp: data_split}

        aux_utils.write_yaml(data_splits, data_split_file)

    def record_model_meta(self):
        """
        Regroups and records important metadata related to training model with the model in
        the 'train_metedata.yml' file.
        """
        model_metadata = {}
        model_metadata["training"] = self.training_config
        model_metadata["dataset"] = self.dataset_config
        model_metadata["model"] = self.network_config
        model_metadata["zarr_dir"] = self.zarr_dir

        model_meta_filename = os.path.join(self.save_folder, "model_metadata.yml")
        aux_utils.write_yaml(model_metadata, model_meta_filename)

    def train(self):
        """
        Run training loop for model, according to parameters set in self.network_config.

        Dataloaders and models must already be initatied. Training results and progress are saved
        in save_dir specified in 'training' section of torch_config each time a test is run.
        """
        assert self.train_dataloader and self.test_dataloader and self.val_dataloader, (
            "Dataloaders " " must be initated. Try 'object_name'.generate_dataloaders()"
        )
        assert self.model, "Model must be initiated. Try 'object_name'.load_model()"

        # init io and saving
        start = time.time()
        self.writer = SummaryWriter(log_dir=self.save_folder)

        # init optimizer and lr regularization
        self.model.train()
        self.optimizer = self.optimizer(
            self.model.parameters(), lr=self.training_config["learning_rate"]
        )
        self.scheduler = self.scheduler(
            self.optimizer, patience=3, mode="min", factor=0.5
        )
        # TODO: make this parameter configurable
        self.early_stopper = EarlyStopping(
            path=self.save_folder,
            patience=3,
            verbose=False,
        )

        # train
        train_loss_list = []
        val_loss_list = []
        test_loss_list = []
        for i in range(self.training_config["epochs"]):
            # Setup epoch
            epoch_time = time.time()
            train_loss = 0
            print(f"Epoch {i}:")

            for current, minibatch in enumerate(self.train_dataloader):

                # pretty printing
                io_utils.show_progress_bar(self.train_dataloader, current)

                # get sample and target (remember we remove the extra batch dimension)
                input_ = minibatch[0][0].cuda(device=self.device).float()
                target_ = minibatch[1][0].cuda(device=self.device).float()

                # run through model
                output = self.model(input_, validate_input=True)
                loss = self.criterion(output, target_)
                train_loss += loss.item()

                # optimize on weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            train_loss_list.append(train_loss / self.train_dataloader.__len__())

            # run validation and loss scheduler every 'testing_stride' epochs
            val_loss = self.run_test(validate_mode=True)
            val_loss_list.append(val_loss)
            self.scheduler.step(val_loss)
            self.early_stopper(val_loss=val_loss, model=self.model, epoch=i)

            # save model checkpoint every 'save_model_stride' epochs
            if (
                i % self.training_config["save_model_stride"] == 0
                or i == self.training_config["epochs"] - 1
            ):
                self.save_model(i, val_loss, input_)

            # send epoch summary to stdout
            print(f"\t Training loss: {train_loss_list[-1]}")
            if i % 1 == 0:
                print(f"\t Validation loss: {val_loss_list[-1]}")
            print(
                f"\t Epoch time: {time.time() - epoch_time}, Total_time: {time.time() - start}"
            )
            print(" ")

            # save to tensorboard
            self.writer.add_scalar(
                tag="train_loss", scalar_value=train_loss_list[-1], global_step=i
            )
            self.writer.add_scalar(
                tag="validation_loss", scalar_value=val_loss_list[-1], global_step=i
            )

            if self.early_stopper.early_stop:
                print("\t Stopping early...")
                break

        # save loss figures (overwrites previous)
        print(f"\t Training complete. Time taken: {time.time()-start}")
        print(
            f"\t Training results and testing predictions saved at: \n\t\t{self.save_folder}"
        )

        self.writer.close()

    def run_test(self, epoch=0, validate_mode=False, to_writer=True):
        """
        Runs test on all samples in a test_dataloader. Equivalent to one epoch on test/val data
        without updating weights. Runs metrics on the test results (given in criterion) and saves
        the results in a save folder, if specified.

        Assumes that all tensors are on the GPU. If not, tensor devices can be specified through
        'device' parameter in torch config.

        :param int epoch: training epoch test was run at
        :param bool validate_mode: run in validation mode to just produce loss (for lr scheduler)
        :param bool to_writer: Whether to record figures to the active SummaryWriter logging this
                            session. If false, will save figures as png instead.
        :return float avg_loss: average testing loss per sample of given data set
        """
        # set the model to evaluation mode
        self.model.eval()

        # Calculate the loss on the images in the test set
        cycle_loss = 0
        samples = []
        targets = []
        outputs = []

        # determine data source
        if validate_mode:
            dataloader = self.val_dataloader
            process = "running loss scheduler"
        else:
            dataloader = self.test_dataloader
            process = "testing"

        for current, minibatch in enumerate(dataloader):
            io_utils.show_progress_bar(dataloader, current, process=process)

            # get input/target
            input_ = minibatch[0][0].to(self.device).float()
            target_ = minibatch[1][0].to(self.device).float()
            sample, target = input_, target_

            # run through model
            output = self.model(input_, validate_input=True)
            loss = self.criterion(output, target_)
            cycle_loss += loss.item()

            # save filters (remember to take off gpu)
            rem = lambda x: x.detach().cpu().numpy()
            if current < 1:
                samples.append(rem(sample))
                targets.append(rem(target))
                outputs.append(rem(output))

        # save test figures
        # TODO: This is too long, move to auxilary function
        arch = self.network_config["architecture"]
        fig, ax = plt.subplots(1, 3, figsize=(18, 6))
        try:
            ax[0].imshow(
                np.mean(samples.pop(), 2)[0, 0]
                if arch == "2.5D"
                else samples.pop()[0, 0],
                cmap="gray",
            )
            ax[0].set_title("input image")
        except TypeError as e:
            print(
                f"Caught error showing phase input, arguments: {e.args}. "
                "Not saving visualization for this epoch."
            )
        try:
            ax[1].imshow(
                targets.pop()[0, 0, 0] if arch == "2.5D" else targets.pop()[0, 0]
            )
            ax[1].set_title("target")
        except TypeError as e:
            print(
                f"Caught error showing fluorescent target, arguments: {e.args}. "
                "Not saving visualization for this epoch."
            )
        try:
            ax[2].imshow(
                outputs.pop()[0, 0, 0] if arch == "2.5D" else outputs.pop()[0, 0]
            )
            ax[2].set_title("prediction")
        except TypeError as e:
            print(
                f"Caught error showing prediction, arguments: {e.args}. "
                "Not saving visualization for this epoch."
            )
        try:
            for i in range(3):
                ax[i].axis("off")
            plt.tight_layout()
            if to_writer:
                self.writer.add_figure(
                    tag="predictions",
                    figure=fig,
                    global_step=epoch,
                    close=False,
                )
            else:
                plt.savefig(
                    os.path.join(self.save_folder, f"prediction_epoch_{epoch}.png")
                )
            if self.plot:
                plt.show()
        except Exception as e:
            print(
                f"Caught error plotting visualization figure, arguments: {e.args}. "
                "Not saving visualization for this epoch."
            )
        plt.close()

        # set back to training mode
        self.model.train()

        # return average loss
        avg_loss = cycle_loss / dataloader.__len__()
        return avg_loss

    def save_model(self, epoch, avg_loss, sample):
        """
        Utility function for saving pytorch model after a test cycle. Parameters are used directly
        in test cycle.

        :param int epoch: see name
        :param float avg_loss: average loss of each cample in testing cycle at epoch 'epoch'
        :param torch.tensor sample: sample input to model (for tensorboard creation)
        """
        # write tensorboard graph
        if isinstance(sample, torch.Tensor):
            self.writer.add_graph(self.model, sample.to(self.device))
        else:
            self.writer.add_graph(
                self.model, torch.tensor(sample, dtype=torch.float32).to(self.device)
            )

        # save model
        save_file = str(f"saved_model_ep_{epoch}_valloss_{avg_loss:.4f}.pt")
        torch.save(self.model.state_dict(), os.path.join(self.save_folder, save_file))


class EarlyStopping:
    def __init__(
        self,
        path,
        patience=7,
        verbose=False,
        delta=0,
        trace_func=print,
        save_model=True,
    ):
        """
        Early stops the training if validation loss doesn't improve after a given patience.
        Adapted from:
            https://github.com/Bjarten/early-stopping-pytorch

        :param int patience: How long to wait after last time validation loss improved.
                            Default: 7
        :param bool verbose: If True, prints a message for each validation loss improvement.
                            Default: False
        :param float delta: Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        :param str path: Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
        :param funct trace_func: trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.trace_func = trace_func
        self.path = path
        self.save_model = save_model

    def __call__(self, val_loss, model, epoch):
        """
        Determine whether stopping is necessary this epoch.
        Should be called every epoch to enforce early stopping.

        :param float val_loss: avg loss from validation dataset
        :param nn.Module model: model from which to save early
        :param int epoch: current epoch at time of call
        """
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            save_here = True
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(
                f"EarlyStopping counter: {self.counter} out of {self.patience}"
            )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

        if self.early_stop:
            self.save_checkpoint(val_loss, model, epoch)

    def save_checkpoint(self, val_loss, model, epoch):
        """
        Saves model when validation loss decrease.

        :param float val_loss: avg loss from validation dataset
        :type nn.Module model: model from which to save early
        """
        if self.save_model == False:
            return
        if self.verbose:
            self.trace_func(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )

        save_file = str(f"early_stop_model_ep_{epoch}_testloss_{val_loss:.4f}.pt")
        torch.save(model.state_dict(), os.path.join(self.path, save_file))

        self.val_loss_min = val_loss
