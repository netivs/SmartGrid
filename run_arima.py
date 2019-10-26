import argparse
import os
import sys
import tensorflow as tf
import yaml
from model.arima import Arima

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

def print_arima_info(mode, config):
    print('----------------------- INFO -----------------------')

    print('|--- ALG:\t{}'.format(config['alg']))
    print('|--- BASE_DIR:\t{}'.format(config['base_dir']))
    print('|--- LOG_LEVEL:\t{}'.format(config['log_level']))
    print('|--- GPU:\t{}'.format(config['gpu']))

    print('----------------------- DATA -----------------------')
    print('|--- BATCH_SIZE:\t{}'.format(config['data']['batch_size']))
    print('|--- RAW_DATASET_DIR:\t{}'.format(config['data']['raw_dataset_dir']))
    print('|--- LEN_DATA:\t{}'.format(config['data']['len_data']))

    print('----------------------- MODEL -----------------------')
    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- VERIFIED_PERCENTAGE:\t{}'.format(config['model']['verified_percentage']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))


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
    if infor_correct != 'y' and infor_correct != 'yes':
        raise RuntimeError('Information is not correct!')

def evaluate_arima(config):
    pass

def test_arima(config):
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    model = Arima(**config)
    model.test()

if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config-file', default='config/config_arima.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='test', type=str,
                        help='Run mode.')
    parser.add_argument('--output_filename', default='data/arima_predictions.npz')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print_arima_info(args.mode, config)

    if args.mode == 'test':
        test_arima(config)
    elif args.mode == 'evaluate' or args.mode == 'evaluation':
        evaluate_arima(config)
    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
