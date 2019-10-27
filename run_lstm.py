import argparse
import os
import sys
import tensorflow as tf
import yaml
from model.encoder_decoder_supervisor import EncoderDecoder

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def print_lstm_info(mode, config):
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
    print('|--- LEN_DATA:\t{}'.format(config['data']['len_data']))

    print('----------------------- MODEL -----------------------')
    print('|--- MODEL_TYPE:\t{}'.format(config['model']['model_type']))
    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- VERIFIED_PERCENTAGE:\t{}'.format(config['model']['verified_percentage']))
    print('|--- L1_DECAY:\t{}'.format(config['model']['l1_decay']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))
    print('|--- N_RNN_LAYERS:\t{}'.format(config['model']['n_rnn_layers']))


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


def train_lstm_ed(config):
    print('|-- Run model training dgc_lstm.')
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    model = EncoderDecoder(is_training=True, **config)
    model.plot_models()
    model.train()


def test_lstm_ed(config):
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    model = EncoderDecoder(is_training=False, **config)
    model.test()


def evaluate_lstm_ed(config):
    # with tf.device('/device:GPU:{}'.format(config['gpu'])):
    model = EncoderDecoder(is_training=False, **config)
    model.evaluate()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config-file', default='config/config_lstm_ed.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--model', default='lstm', type=str,
                        help='model.')
    parser.add_argument('--output_filename', default='data/lstm_ed_predictions.npz')
    args = parser.parse_args()

    with open(args.config_file) as f:
        config = yaml.load(f)

    print_lstm_info(args.mode, config)

    if args.mode == 'train':
        if config['model']['model_type'] == 'ed':
            train_lstm_ed(config)
        else:
            raise RuntimeError('|--- Model should be lstm or ed (encoder-decoder)!')
    elif args.mode == 'evaluate' or args.mode == 'evaluation':

        if config['model']['model_type'] == 'ed':
            evaluate_lstm_ed(config)
        else:
            raise RuntimeError('|--- Model should be lstm or ed (encoder-decoder)!')

    elif args.mode == "test":

        if config['model']['model_type'] == 'ed':
            test_lstm_ed(config)
        else:
            raise RuntimeError('|--- Model should be lstm or ed (encoder-decoder)!')

    else:
        raise RuntimeError("Mode needs to be train/evaluate/test!")
