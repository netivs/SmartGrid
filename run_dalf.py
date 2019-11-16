import argparse
import os
import sys
import tensorflow as tf
import yaml
from model.dalf_supervisor import DALFSupervisor
from lib import utils

config = tf.ConfigProto()
session = tf.Session(config=config)

def print_dalf_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- MODE:\t{}'.format(mode))
    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))
    print('|--- LOG_LEVEL:\t{}'.format(config['log_level']))
    print('|--- GPU:\t{}'.format(config['gpu']))

    print('----------------------- DATA -----------------------')
    print('|--- BATCH_SIZE:\t{}'.format(config['data']['batch_size']))
    print('|--- RAW_DATASET_DIR:\t{}'.format(config['data']['raw_dataset_dir']))
    print('|--- EVAL_BATCH_SIZE:\t{}'.format(config['data']['eval_batch_size']))
    print('|--- TEST_BATCH_SIZE:\t{}'.format(config['data']['test_batch_size']))
    print('|--- PERCENT_TEST_DATA:\t{}'.format(config['data']['percent_test_data']))

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

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- OPTIMIZER:\t{}'.format(config['train']['optimizer']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- CONTINUE_TRAIN:\t{}'.format(config['train']['continue_train']))

    else:
        print('----------------------- TEST -----------------------')
        print('|--- RUN_TIMES:\t{}'.format(config['test']['run_times']))

    print('----------------------------------------------------')
    infor_correct = input('Is the information correct? y(Yes)/n(No):')
    if infor_correct != 'y' and infor_correct != 'yes' and infor_correct != 'YES' and infor_correct != 'Y':
        raise RuntimeError('Information is not correct!')


def train_dalf(config):
    dalf_supervisor = DALFSupervisor(is_training=True, **config)
    dalf_supervisor.plot_models()
    dalf_supervisor.train()


def test_dalf(config):
    dalf_supervisor = DALFSupervisor(is_training=False, **config)
    dalf_supervisor.test()
    dalf_supervisor.plot_series()


def evaluate_dalf(config):
    dalf_supervisor = DALFSupervisor(is_training=False, **config)
    dalf_supervisor.evaluate()

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_file', default='config/config_dalf.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--output_filename', default='data/dalf_predictions.npz')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print_dalf_info(args.mode, config)

    if args.mode == 'train':
        train_dalf(config)

    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_dalf(config)

    elif args.mode == "test":
        test_dalf(config)
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
