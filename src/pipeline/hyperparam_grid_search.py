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
    return 0

def gen_hyperparam_grid() -> int:
    '''Generate a list of namedtuples
    of hyperparams to evaluate
    '''
    return 5

def train_model(hyperparam_idx: int, retcode_download: int, N_gridsize: int) -> int:
    '''Look up hyperparams from store
    and train model
    '''
    

    def evaluate_model(model_idx: int):
        '''Evaluate model on test set
        and store result
        '''
        pass

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
    #with kfp.dsl.ParallelFor(list(range(10))) as i:
    for i in list(range(10)): #try to be as close as possible to canonical python for DS people
        retcode_model = train_model_op(i, retcode_download.output, N_gridsize.output)
        retcode_list.append(retcode_model)

    #for i in range(N_gridsize.output):
    #    retcode_model = train_model_op(i, retcode_download.output, N_gridsize.output)

    best_idx = find_best_op().after(*retcode_list)


#kfp.compiler.Compiler().compile(run_all, 'run_all.zip')
from kfp_tekton.compiler import TektonCompiler
TektonCompiler().compile(run_pipeline, 'hyperparam_skeleton.yaml')
