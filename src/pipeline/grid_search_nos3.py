import utils, config

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import List, Tuple

import kfp
from kfp.components import func_to_container_op
import kfp_tekton

BASE_IMAGE = config.BASE_IMAGE
S3_END_POINT = config.S3_END_POINT
S3_ACCESS_ID = config.S3_ACCESS_ID
S3_ACCESS_KEY = config.S3_ACCESS_KEY
bucket_name = config.BUCKET_NAME

get_client = utils.get_client
create_bucket = utils.create_bucket
read_from_store = utils.read_from_store
write_to_store = utils.write_to_store

class Net(nn.Module):
    def __init__(self, n_inputs=2, n_outputs=1, n_hidden_nodes=10, n_hidden_layers=1, activation=nn.ReLU(), output_activation=None):
        super(Net, self).__init__()

        self.layer_list = nn.ModuleList()

        for i in range(n_hidden_layers):
            if i==0:
                self.layer_list.append(nn.Linear(n_inputs, n_hidden_nodes))
            else:
                self.layer_list.append(nn.Linear(n_hidden_nodes, n_hidden_nodes))
        
        self.output_layer = nn.Linear(n_hidden_nodes, n_outputs)

        self.activation = activation
        self.output_activation = output_activation

    def forward(self, x):
        out = x

        for layer in self.layer_list:
            out = self.activation(layer(out))

        out = self.output_layer(out)
        if self.output_activation is not None:
            out = self.output_activation(out)

        return out



def download_data() -> int:
    '''Download and store data in persistent storage
    '''

    def generate_binary_data(N_examples=1000, seed=None):
    #Generate N_examples points with two features each
    #
    #Args:
    #    seed: seed that should be fixed if want to generate same points again    
    #Returns:
    #    features: A 2-dimensional numpy array with one row per example and one column per feature
    #    target: A 1-dimensional numpy array with one row per example denoting the class - 0 or 1

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

    return features_train, target_train, features_test, target_test

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

    return grid

def train_model(hyperparam_idx: int) -> []:
    '''Look up hyperparams from store
    and train model
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Device = {device}')

    features_train, target_train, features_test, target_test = download_data()
    
    features_train = torch.from_numpy(features_train).float()
    target_train = torch.from_numpy(target_train).float()
    features_test = torch.from_numpy(features_test).float()
    target_test = torch.from_numpy(target_test).float()

    grid = gen_hyperparam_grid()

    conf = grid[hyperparam_idx]

    lr = float(conf.get('lr', 1e-2))
    N_epochs = int(conf.get('N_epochs', 10000))
    num_hidden_layers = int(conf.get('num_hidden_layers', 1))
    num_nodes = int(conf.get('num_nodes', 2))
    activation = conf.get('activation', 'relu')

    #should be dependent on vars read from config
    if activation=='relu':
        activation = nn.ReLU()
    elif activation=='sigmoid':
        activation = nn.Sigmoid()

    model = Net(n_inputs=2, n_outputs=1, n_hidden_nodes=num_nodes, n_hidden_layers=num_hidden_layers, activation=activation, output_activation=nn.Sigmoid())
    
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

    def evaluate_model(model, features_test, target_test):
        '''Evaluate model on test set
        and store result
        '''
        model.eval()

        if device!='cpu':
            features_test = features_test.to(device)
            target_test = target_test.to(device)

        out = model(features_test)
        loss = criterion(out.squeeze(), target_test.squeeze())
        

        return loss

    test_loss = evaluate_model(model, features_test, target_test)
    print(f'Test  Loss : {test_loss}')


    return [hyperparam_idx, test_loss.item()]

def find_best(data: List[Tuple[int, float]]):
    grid = gen_hyperparam_grid()

    print(data)
    for elem in data:
        print(elem)

    idx = np.argmax([d[1] for d in data])

    print(f'Best: {grid[idx]} Error: {data[idx][1]}')


train_model_op = func_to_container_op(train_model, base_image=BASE_IMAGE, packages_to_install=None, modules_to_capture=["utils"], use_code_pickling=True)
find_best_op = func_to_container_op(find_best, base_image=BASE_IMAGE, packages_to_install=None, modules_to_capture=["utils"], use_code_pickling=True)

@kfp.dsl.pipeline(
    name='Full pipeline'
)
def run_pipeline():
    #retcode_download = download_data_op()
    
    #N_gridsize = gen_hyperparam_grid_op()

    retcode_list = [] #will adapt this later - parallelfor?    
    for i in list(range(10)): #try to be as close as possible to canonical python for DS people
        retcode_model = train_model_op(i)
        retcode_list.append(retcode_model.output)

    find_best_op(retcode_list)

from kfp_tekton.compiler import TektonCompiler
TektonCompiler().compile(run_pipeline, 'grid_search_nos3.yaml')
