import micro_dl.torch_unet.networks.Unet25D as Unet25D
import micro_dl.torch_unet.networks.Unet2D as Unet2D

def model_init(network_config):
    '''
    initializes network model from a configuration file
    
    Params:
        - network_config -> dictionary: dict containing the configuration parameters for the model
    '''
    if network_config['architecture'] == '2.5D':
        model = Unet25D.Unet25d(in_channels = network_config['in_channels'],
                                out_channels = network_config['out_channels'], 
                                residual = network_config['residual'],
                                task = network_config['task'])
    elif network_config['architecture'] == '2D':
        model = Unet2D.Unet2d(in_channels = network_config['in_channels'],
                                out_channels = network_config['out_channels'], 
                                residual = network_config['residual'],
                                task = network_config['task'])
    else:
        raise NotImplementedError('Only 2.5D and 2D architectures available.')
    model.cuda()
    
    return model


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
    