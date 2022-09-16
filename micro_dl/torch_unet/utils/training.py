import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

import os
import time
from torch.utils.tensorboard import SummaryWriter
import datetime

import micro_dl.torch_unet.utils.model as model_utils
import micro_dl.torch_unet.utils.dataset as ds
import micro_dl.torch_unet.utils.io as io_utils


class TorchTrainer():
    '''
    TorchTrainer object which handles all the procedures involved with training a pytorch model.
    The trainer uses a the model.py and dataset.py utility modules to instantiate and load a model and
    training data by passing them through the existing tensorflow dataset creators and reformatting
    the outputs.
    
    Functionality of the class can be achieved without specifying full torch_config. However, full
    functionality requires full configuration file.
    '''
    def __init__(self, torch_config):
        self.torch_config = torch_config
        self.network_config = self.torch_config['model']
        self.training_config = self.torch_config['training']
        
        self.train_dataloader = None
        self.test_dataloader = None
        self.val_dataloader = None
        
        self.model = None
        
        #init token specific parameters - optimizer, loss, device
        #optimizer
        assert self.training_config['optimizer'] in {'adam', 'sgd'}, 'optimizer must be \'adam\' or \'sgd\''
        if self.training_config['optimizer'] == 'adam':
            self.optimizer = optim.Adam
        else:
            self.optimizer = optim.SGD
        #loss
        assert self.training_config['loss'] in {'mse', 'l1', 'cossim','cel'}, f'loss not supported. Try one of' \
            '\'mse\', \'mae\', \'cossim\', \'l1\''
        if self.training_config['loss'] == 'mse':
            self.criterion = nn.MSELoss()
        elif self.training_config['loss'] == 'l1':
            self.criterion = nn.L1Loss()
        elif self.training_config['loss'] == 'cossim':
            self.criterion = nn.CosineSimilarity()
        elif self.training_config['loss'] == 'cel':
            self.criterion = nn.CrossEntropyLoss()
        #device
        assert self.training_config['device'] in {'cpu','gpu'}, 'device must be cpu or gpu'
        if self.training_config['device'] == 'gpu':
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        #plotting
        self.plot = False
    
    def load_model(self, init_dir = False) -> None:
        '''
        Initializes a model according to the network configuration dictionary used to train it, and, if provided,
        loads the parameters saved in init_dir into the model's state dict.
        
        :param str init_dir: directory containing model weights and biases
        '''
        assert self.network_config != None, 'Network configuration must be initiated to load model'
        model = model_utils.model_init(self.network_config)
        
        if init_dir:
            model_dir = self.network_config['model_dir']
            readout = model.load_state_dict(torch.load(model_dir))
            print(readout)
        self.model = model
        if self.training_config['device'] == 'gpu':
            self.model.cuda()
        
    def generate_dataloaders(self) -> None:
        '''
        Helper that generates train, test, validation torch dataloaders for loading samples
        into network for training and testing. 
        
        Dataloaders are set to class variables. torch_config can be specified by parent class initiation.
        If specified in initiation, config provided in call will be prioritized.
        
        :param pd.dataframe torch_config: configuration dataframe with model & training init parameters
        '''
        assert self.torch_config != None, 'torch_config must be specified in object' \
            'initiation '
            
        #determine transforms/augmentations
        transforms = [ds.ToTensor()]
        target_transforms = [ds.ToTensor()]
        if self.training_config['mask']:
            target_transforms.append(ds.GenerateMasks(self.training_config['mask_type']))
        
        #init dataset container and pull dataset split objects
        torch_data_container = ds.TorchDataset(self.torch_config['train_config_path'],
                                            transforms=transforms,
                                            target_transforms=target_transforms)
        train_dataset = torch_data_container['train']
        test_dataset = torch_data_container['test']
        val_dataset = torch_data_container['val']
        
        #init dataloaders
        self.train_dataloader = torch.utils.data.DataLoader(dataset = train_dataset, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset = test_dataset, shuffle=True)
        self.val_dataloader = torch.utils.data.DataLoader(dataset = val_dataset, shuffle=True)
    
    def get_save_location(self):
        '''
        Initates save folder if not already initated. 
        Directory is named depending on time of training. All training/testing information is saved to this
        directory.
        '''
        now = str(datetime.datetime.now()).replace(' ', '_').replace(':','_').replace('-','_')[:-10]
        self.save_folder = os.path.join(self.training_config['save_dir'], f'training_model_{now}')
        if not os.path.exists(self.save_folder):
            os.makedirs(self.save_folder)
    
    def train(self):
        '''
        Run training loop for model, according to parameters set in self.network_config.
        
        Dataloaders and models must already be initatied. Training results and progress are saved in 
        save_dir specified in 'training' section of torch_config each time a test is run.
        '''
        assert self.train_dataloader and self.test_dataloader and self.val_dataloader, 'Dataloaders must be initated. ' \
            'Try \'object_name\'.generate_dataloaders()'
        assert self.model, 'Model must be initiated. Try \'object_name\'.load_model()'
        
        #init io and saving
        start = time.time()
        self.get_save_location()
        self.writer = SummaryWriter(log_dir = self.save_folder)
        
        #init optimizer
        self.model.train()
        self.optimizer = self.optimizer(self.model.parameters(), lr = self.training_config['learning_rate'])
        
        #train
        train_loss_list = []
        test_loss_list = []
        for i in range(self.training_config['epochs']):
            print(f'Epoch {i}:')
            epoch_time = time.time()
            train_loss = 0
            for current, minibatch in enumerate(self.train_dataloader): 
                #pretty printing
                io_utils.show_progress_bar(self.train_dataloader, current)
                
                #get sample and target (remember we remove the extra batch dimension)
                input_ = minibatch[0][0].to(self.device).float()
                target_ = minibatch[1][0].to(self.device).float()
                mask = minibatch[2][0].to(self.device).float()
                
                #if specified mask sample to get input and target
                #TODO: change caching to include masked inputs since masks never change
                if self.training_config['mask']:
                    input_ = torch.mul(input_, mask)
                    target_ = torch.mul(target_, mask)

                #run through model
                output = self.model(input_)
                loss = self.criterion(output, target_)
                train_loss += loss.item()
                
                #optimize on weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
            train_loss_list.append(train_loss/self.train_dataloader.__len__())
            
            #run testing cycle every 'testing_stride' epochs
            if i % self.training_config['testing_stride'] == 0:
                test_loss = self.run_test(current)
                test_loss_list.append(test_loss/self.test_dataloader.__len__())
                
            # save model every 'save_model_stride' epochs
            if i % self.training_config['save_model_stride'] == 0:
                self.save_model(current, test_loss, input_)
                
            print(f'\t Training loss: {train_loss_list[-1]}')
            if i % 1 == 0:
                print(f'\t Testing loss: {test_loss_list[-1]}')
            print(f'\t Epoch time: {time.time() - epoch_time}, Total_time: {time.time() - start}')
            print(' ')
        
        #save loss figures
        print(f'\t Training complete. Time taken: {time.time()-start}')
        print(f'\t Training results and testing predictions saved at: \n\t\t{self.save_folder}')
        fig = plt.figure(figsize = (14,7))
        plt.plot(train_loss_list, label = 'training loss')
        plt.plot(test_loss_list, label = 'testing loss')
        plt.legend()
        plt.savefig(os.path.join(self.save_folder, 'training_loss.png'), bbox_inches = 'tight')
        plt.cla()

        self.writer.close()
    
    def run_test(self, epoch, mask_override = False):
        '''
        Runs test on all samples in a test_dataloader. Equivalent to one epoch on test data without updating weights.
        Runs metrics on the test results (given in criterion) and saves the results in a save folder, if specified.
        
        Assumes that all tensors are on the GPU. If not, tensor devices can be specified through 'device' parameter.

        :param int epoch: training epoch test was run at
        :param str mask_override: overrides the masking parameter for testing
        :return float avg_loss: average testing loss per sample of entire test set
        '''
        #set the model to evaluation mode
        self.model.eval()
        
        # Calculate the loss on the images in the test set
        test_loss = []
        samples = []
        targets = []
        outputs = []
        for current, minibatch in enumerate(self.test_dataloader):
            #pretty printing
            io_utils.show_progress_bar(self.test_dataloader, current, process = 'testing')
            
            #get input/target
            input_ = minibatch[0][0].to(self.device).float()
            target_ = minibatch[1][0].to(self.device).float()
            sample, target = input_, target_
            
            # if mask provided, mask sample to get input and target
            if mask_override:
                mask_ = minibatch[2][0].to(self.device).float()
                input_ = torch.mul(input_, mask_)
                target_ = torch.mul(target_, mask_)
            
            #run through model
            output = self.model(input_)
            loss = self.criterion(output, target_)
            test_loss.append(loss.item())
            
            #save filters (remember to take off gpu)
            rem = lambda x: x.detach().cpu().numpy()
            if current < 1:
                samples.append(rem(sample))
                targets.append(rem(target))
                outputs.append(rem(output))
        
        #get avg loss per sample from total loss
        avg_loss = np.sum(test_loss)/self.test_dataloader.__len__()
        
        #save test figures
        fig, ax = plt.subplots(1,3,figsize = (18,6))
        ax[0].imshow(np.mean(samples.pop(), 2)[0,0], cmap = 'gray')
        ax[0].set_title('mean input phase image')
        ax[1].imshow(targets.pop()[0,0,0])
        ax[1].set_title('target')
        ax[2].set_title('prediction')
        ax[2].imshow(outputs.pop()[0,0,0])
        for i in range(3):
            ax[i].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_folder, f'prediction_epoch_{epoch}.png'))
        if self.plot:
            plt.show()
        
        #set back to training mode
        self.model.train()
        return avg_loss 

    def save_model(self, epoch, avg_loss, sample):
        '''
        Utility function for saving pytorch model after a test cycle. Parameters are used directly in test cycle.
        
        Params:
            - epoch -> int: see name
            - avg_loss -> float: average loss of each cample in testing cycle at epoch 'epoch'
            - sample -> torch.tensor: sample input to model (for tensorboard creation)
        '''
        #write tensorboard graph
        if isinstance(sample, torch.Tensor):
            self.writer.add_graph(self.model, sample.to(self.device))
        else:
            self.writer.add_graph(self.model, torch.tensor(sample, dtype=torch.float32).to(self.device))
        
        #save model
        save_file = str(f'saved_model_ep_{epoch}_testloss_{avg_loss:.4f}.pt')
        torch.save(self.model.state_dict(), os.path.join(self.save_folder, save_file))