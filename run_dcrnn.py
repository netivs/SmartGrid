import argparse
import os
import sys
import tensorflow as tf
import yaml
from model.dcrnn_supervisor import DCRNNSupervisor
from lib import utils

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

def print_dcrnn_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))
    print('|--- LOG_LEVEL:\t{}'.format(config['log_level']))
    print('|--- GPU:\t{}'.format(config['gpu']))

    print('----------------------- DATA -----------------------')
    print('|--- BATCH_SIZE:\t{}'.format(config['data']['batch_size']))
    print('|--- RAW_DATASET_DIR:\t{}'.format(config['data']['raw_dataset_dir']))
    print('|--- EAL_BATCH_SIZE:\t{}'.format(config['data']['val_batch_size']))
    print('|--- TEST_BATCH_SIZE:\t{}'.format(config['data']['test_batch_size']))
    print('|--- LEN_DATA:\t{}'.format(config['data']['len_data']))

    print('----------------------- MODEL -----------------------')
    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- VERIFIED_PERCENTAGE:\t{}'.format(config['model']['verified_percentage']))
    print('|--- L1_DECAY:\t{}'.format(config['model']['l1_decay']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))
    print('|--- NUM_RNN_LAYERS:\t{}'.format(config['model']['num_rnn_layers']))
    print('|--- FILTER_TYPE:\t{}'.format(config['model']['filter_type']))
    print('|--- MAX_DIFFUSION_STEP:\t{}'.format(config['model']['max_diffusion_step']))
    print('|--- USE_CURRICULUMN_LEARNING:\t{}'.format(config['model']['use_curriculum_learning']))


    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- BASE_LR:\t{}'.format(config['train']['base_lr']))
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPOCH:\t{}'.format(config['train']['epoch']))
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- EPSILON:\t{}'.format(config['train']['epsilon']))
        print('|--- GLOBAL_STEP:\t{}'.format(config['train']['global_step']))
        print('|--- LR_DECAY_RATIO:\t{}'.format(config['train']['lr_decay_ratio']))
        print('|--- MAX_GRAD_NORM:\t{}'.format(config['train']['max_grad_norm']))
        print('|--- MAX_TO_KEEP:\t{}'.format(config['train']['max_to_keep']))
        print('|--- MIN_LEARNING_RATE:\t{}'.format(config['train']['min_learning_rate']))
        print('|--- OPTIMIZER:\t{}'.format(config['train']['optimizer']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- STEPS:\t{}'.format(config['train']['steps']))
        print('|--- TEST_EVERY_N_EPOCHS:\t{}'.format(config['train']['test_every_n_epochs']))

    else:
        print('----------------------- TEST -----------------------')
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')


def train_dcrnn(adj_mx, config):
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    dcrnn_supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
    dcrnn_supervisor.train(sess = session)


def test_dcrnn(adj_mx, config):
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    # dcrnn_supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
    # dcrnn_supervisor.test(sess = session)
    pass


def evaluate_dcrnn(adj_mx, config):
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    dcrnn_supervisor = DCRNNSupervisor(adj_mx=adj_mx, **config)
    dcrnn_supervisor.evaluate(sess = session)


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default='config/config_dcrnn.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    graph_pkl_filename = config['data'].get('graph_pkl_filename')
    adj_mx = utils.load_graph_data(graph_pkl_filename)
    print_dcrnn_info(args.mode, config)

    if args.mode == 'train':
        train_dcrnn(adj_mx, config)

    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_dcrnn(adj_mx, config)

    elif args.mode == "test":
        test_dcrnn(adj_mx, config)
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
