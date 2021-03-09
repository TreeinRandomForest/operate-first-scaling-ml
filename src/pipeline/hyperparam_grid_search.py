import utils, config

import kfp
from kfp.components import func_to_container_op
import kfp_tekton

BASE_IMAGE = config.BASE_IMAGE
S3_END_POINT = config.S3_END_POINT
S3_ACCESS_ID = config.S3_ACCESS_ID
S3_ACCESS_KEY = config.S3_ACCESS_KEY

# Note: this is very specific to the type of model
# and metric. Can generalize to arbitrary models later.

#Using return values to structure DAG

def download_data() -> int:
    '''Download and store data in persistent storage
    '''
    import numpy as np
    from .config import *
    from .utils import *    

    def generate_binary_data(N_examples=1000, seed=None):
    '''Generate N_examples points with two features each
    
    Args:
        seed: seed that should be fixed if want to generate same points again
    
    Returns:
        features: A 2-dimensional numpy array with one row per example and one column per feature
        target: A 1-dimensional numpy array with one row per example denoting the class - 0 or 1
    '''

        if seed is not None:
            np.random.seed(seed)

        features = []
        target = []

        for i in range(N_examples):
            #class = 0
            r = np.random.uniform() #class 0 has radius between 0 and 1
            theta = np.random.uniform(0, 2*np.pi) #class 0 has any angle between 0 and 360 degrees

            features.append([r*np.cos(theta), r*np.sin(theta)])
            target.append(0)

            #class = 1
            r = 3 + np.random.uniform() #class 1 has radius between 3+0=3 and 3+1=4
            theta = np.random.uniform(0, 2*np.pi) #class 1 has any angle between 0 and 360 degrees

            features.append([r*np.cos(theta), r*np.sin(theta)])
            target.append(1)

        features = np.array(features)
        target = np.array(target)

        return features, target

    features_train, target_train = generate_binary_data(N_examples=1000, seed=100)
    features_test, target_test = generate_binary_data(N_examples=500, seed=105)

    write_to_store(config.BUCKET_NAME, features_train, 'features_train')
    write_to_store(config.BUCKET_NAME, target_train, 'target_train')
    write_to_store(config.BUCKET_NAME, features_test, 'features_test')
    write_to_store(config.BUCKET_NAME, target_test, 'target_test')

    return 0

def gen_hyperparam_grid() -> int:
    '''Generate a list of namedtuples
    of hyperparams to evaluate
    '''
    grid = []
    for num_hidden_layers in [1,2,3]:
        for num_nodes in [1,2,3]:
            for activation in ['relu', 'sigmoid']:
                grid.append({'num_hidden_layers': num_hidden_layers,
                             'num_nodes': num_nodes,
                             'activation': activation
                            })


    write_to_store(config.BUCKET_NAME, grid, 'hyperparam_grid')

    return len(grid)

def train_model(hyperparam_idx: int, retcode_download: int, N_gridsize: int) -> int:
    '''Look up hyperparams from store
    and train model
    '''
    import torch
    import torch.nn as nn
    import torch.optim as optim

    if hyperparam_idx >= N_gridsize:
        raise ValueError("hyperparam_idx cannot be >= N_gridsize")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device}')

    features_train = read_from_store(config.BUCKET_NAME, 'features_train')
    target_train = read_from_store(config.BUCKET_NAME, 'target_train')

    config = read_from_store(config.BUCKET_NAME, 'hyperparam_grid')[hyperparam_idx]
    lr = config.get('lr', 1e-2)
    N_epochs = config.get('N_epochs', 10000)
    num_hidden_layers = config.get('num_hidden_layers', 1)
    num_nodes = config.get('num_nodes', 2)
    activation = config.get('activation', 'relu')

    model = ...

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr) #Adam optimizer
    model.train()    

    if device!='cpu':
        model = model.to(device)
        features_train = features_train.to(device)
        target_train = target_train.to(device)

    for epoch in range(N_epochs): #N_epochs = number of iterations over the full dataset
        features_shuffled = features_train
        target_shuffled = target_train

        out = model(features_shuffled) #predictions from model
        loss = criterion(out.squeeze(), target_shuffled.squeeze()) #loss between predictions and labels

        if epoch % 1000 == 0:
            print(f'epoch = {epoch} loss = {loss}')

        optimizer.zero_grad()
        loss.backward() #compute gradients
        optimizer.step() #update model

    out = model(features_shuffled) #predictions from model
    train_loss = criterion(out.squeeze(), target_shuffled.squeeze()) #loss between predictions and labels
    print(f'Train Loss : {train_loss}')

    def evaluate_model(model):
        '''Evaluate model on test set
        and store result
        '''
        
        features_test = read_from_store(config.BUCKET_NAME, 'features_test')
        target_test = read_from_store(config.BUCKET_NAME, 'target_test')
    
        if device!='cpu':
            features_test = features_test.to(device)
            target_test = target_test.to(device)

        out = model(features_test)
        loss = criterion(out.squeeze(), target_test.squeeze())
        

        return loss

    test_loss = evaluate_model(model)
    print(f'Test  Loss : {test_loss}')

    write_to_store(config.BUCKET_NAME, {'test_loss': test_loss, 'model': model}, f'score_{hyperparam_idx}')

    return hyperparam_idx

def find_best() -> int:
    '''Return idx corresponding
    to best model
    '''
    
    return 10


download_data_op = func_to_container_op(download_data, base_image=BASE_IMAGE)
gen_hyperparam_grid_op = func_to_container_op(gen_hyperparam_grid, base_image=BASE_IMAGE)
train_model_op = func_to_container_op(train_model, base_image=BASE_IMAGE)
find_best_op = func_to_container_op(find_best, base_image=BASE_IMAGE)


@kfp.dsl.pipeline(
    name='Full pipeline'
)
def run_pipeline():
    retcode_download = download_data_op()
    
    N_gridsize = gen_hyperparam_grid_op()

    retcode_list = [] #will adapt this later - parallelfor?
    #with kfp.dsl.ParallelFor(N_gridsize.output) as i: #error: dynamic params are not yet implemented        
    #    retcode_model = train_model_op(i, retcode_download.output, N_gridsize.output)
    #    retcode_list.append(retcode_model)
    
    for i in list(range(10)): #try to be as close as possible to canonical python for DS people
        retcode_model = train_model_op(i, retcode_download.output, N_gridsize.output)
        retcode_list.append(retcode_model)

    #for i in range(N_gridsize.output):
    #    retcode_model = train_model_op(i, retcode_download.output, N_gridsize.output)

    best_idx = find_best_op().after(*retcode_list)


#kfp.compiler.Compiler().compile(run_all, 'run_all.zip')
from kfp_tekton.compiler import TektonCompiler
TektonCompiler().compile(run_pipeline, 'hyperparam_skeleton.yaml')
