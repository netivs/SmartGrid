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
    print('|--- DATA:\t{}'.format(config['data']['data_name']))
    print('|--- GPU:\t{}'.format(config['gpu']))
    print('|--- GENERATE_DATA:\t{}'.format(config['data']['generate_data']))

    print('----------------------- MODEL -----------------------')

    print('|--- SEQ_LEN:\t{}'.format(config['model']['seq_len']))
    print('|--- HORIZON:\t{}'.format(config['model']['horizon']))
    print('|--- INPUT_DIM:\t{}'.format(config['model']['input_dim']))
    print('|--- NUM_NODES:\t{}'.format(config['model']['num_nodes']))
    print('|--- N_RNN_LAYERS:\t{}'.format(config['model']['n_rnn_layers']))
    print('|--- OUTPUT_DIMS:\t{}'.format(config['model']['output_dim']))
    print('|--- RNN_UNITS:\t{}'.format(config['model']['rnn_units']))

    if mode == 'train':
        print('----------------------- TRAIN -----------------------')
        print('|--- EPOCHS:\t{}'.format(config['train']['epochs']))
        print('|--- DROPOUT:\t{}'.format(config['train']['dropout']))
        print('|--- PATIENCE:\t{}'.format(config['train']['patience']))
        print('|--- BATCH:\t{}'.format(config['data']['batch_size']))
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
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        model = EncoderDecoder(is_training=True, **config)
        model.plot_models()
        model.train()


def test_lstm_ed(config):
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        model = EncoderDecoder(is_training=False, **config)
        model.test()


def evaluate_lstm_ed(config):
    with tf.device('/device:GPU:{}'.format(config['gpu'])):
        model = EncoderDecoder(is_training=False, **config)
        model.evaluate()


if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config-file', default='data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--mode', default='train', type=str,
                        help='Run mode.')
    parser.add_argument('--model', default='lstm', type=str,
                        help='model.')
    parser.add_argument('--output_filename', default='data/dcrnn_predictions.npz')
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
