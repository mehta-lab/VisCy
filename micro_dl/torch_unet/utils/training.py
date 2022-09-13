import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import sys
import time

def run_test(test_dataloader, model, criterion, mask = True, plot = True, plot_num = 2, epoch = None,
             save_folder = None, writer = None, device = torch.device('cuda')):
    '''
    Runs test on all samples in a test_dataloader. Equivalent to one epoch on test data without updating weights.
    Runs metrics on the test results (given in criterion) and saves the results in a save folder, if specified.
    
    Assumes that all tensors are on the GPU. If not, tensor devices can be specified through 'device' parameter.
    Params:
        test_dataloader -> Dataloader object: dataloader from which to draw samples
        model -> torch.nn.Module: see name
        criterion -> torch loss function: criterion by which to evaluate performance
        mask -> Boolean: whether to apply mask to inputs and outputs before evaluation
        plot -> Boolean: whether to plot outputs
        plot_num -> int: number of rows to plot, between 2 and test_dataloader.__len__()
        epoch -> int: see name
        save_folder -> str: filepath to save folder
        write -> torch.utils.tensorboard.SummaryWriter: tensorboard summary writer watching model
        device -> torch.device: cpu or gpu 
    '''
    #set the model to evaluation mode
    model.eval()
    
    # Calculate the loss on the images in the test set
    test_loss = []
    samples = []
    targets = []
    outputs = []
    for current, minibatch in enumerate(test_dataloader):
        #pretty printing
        show_progress_bar(test_dataloader, current, process = 'testing')
        
        #get input/target
        input_ = minibatch[0][0].to(device).float()
        target_ = minibatch[1][0].to(device).float()
        sample, target = input_, target_
        
        # if mask provided, mask sample to get input and target
        if mask:
            mask_ = minibatch[2][0].to(device).float()
            input_ = torch.mul(input_, mask)
            target_ = torch.mul(target_, mask)
        
        #run through model
        output = model(input_)
        loss = criterion(output, target_)
        test_loss.append(loss.item())
        
        #save filters (remember to take off gpu)
        rem = lambda x: x.detach().cpu().numpy()
        if current < plot_num:
            samples.append(rem(sample))
            targets.append(rem(target))
            outputs.append(rem(output))
    
    #get avg loss per sample from total loss
    avg_loss = np.sum(test_loss)/test_dataloader.__len__()
    
    if plot:
        fig, ax = plt.subplots(plot_num,7,figsize = (18,3*plot_num))
        for j in range(plot_num):
            sample = samples.pop()
            for i in range(5):
                ax[j][i].imshow(sample[0,0,i], cmap = 'gray')
            ax[j][5].imshow(targets.pop()[0,0,0])
            ax[j][6].imshow(outputs.pop()[0,0,0])
            if j == 0:
                ax[j][2].set_title('input phase images')
                ax[j][5].set_title('target')
                ax[j][6].set_title('prediction')
            for i in range(7):
                ax[j][i].axis('off')
        if epoch:
            plt.suptitle(f'epoch: {epoch}')
        plt.tight_layout()
        plt.show()
    
    if save_folder:
        save_model(model, epoch, save_folder, test_loss, avg_loss, sample, device, fig = fig, writer = writer)
        
    #set model back to train mode
    model.train()
    
    return avg_loss
    

def show_progress_bar(dataloader, current, process = 'training'):
    '''
    Utility function to print tensorflow-like progress bar.
    '''
    current += 1
    bar_length = 50
    fraction_computed = current/dataloader.__len__()
    
    pointer = '>' if fraction_computed < 1 else '='
    loading_string = '='*int(bar_length*fraction_computed) + '>' + '_'*int(bar_length*(1-fraction_computed))
    output_string = f'\t {process} {current}/{dataloader.__len__()} [{loading_string}] ({int(fraction_computed * 100)}%)'
    
    if fraction_computed < 1:
        print(output_string, end='\r')
    else:
        print(output_string)
        
    #for smoother output
    time.sleep(0.2) 

def save_model(model, epoch, save_folder, test_loss, avg_loss, sample, device, fig = None, writer = None, hist = False):
    '''
    Utility function for saving pytorch model after a test cycle. Parameters are used directly in test cycle
    
    Params:
        - model -> Torch.nn.module: training model
        - epoch -> int: see name
        - test_loss -> float: full testing cycle loss
        - save_folder -> filepath (str): see name
        - avg_loss -> float: average loss of each cample in testing cycle
        - fig -> matplotlib.figure: testing comparison images (prediction vs target)
        - sample -> torch.tensor: sample input to model
        - device -> torch.device: device on which to place sample before passing into network.
    '''
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    #write tensorboard graph
    if writer:
        writer.add_graph(model, torch.tensor(sample, dtype=torch.float32).to(device))
    
    #save model
    save_file = str(f'saved_model_ep_{epoch}_testloss_{avg_loss:.4f}.pt')
    torch.save(model.state_dict(), os.path.join(save_folder, save_file))
    
    #save prediction
    if fig:
        f = plt.figure()
        plt.savefig(os.path.join(save_folder, f'prediction_epoch_{epoch}.png'))
        plt.close(f)
    
    #save testloss histogram
    if hist:
        f, ax = plt.subplots(1,1, figsize = (10,5))
        ax.hist(test_loss, bins = 20)
        ax.set_title('test loss histogram')
        plt.savefig(os.path.join(save_folder, f'test_histogram_epoch__{epoch}.png'))
        plt.close(f)