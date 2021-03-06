import os
import time

import keras.callbacks as keras_callbacks
import numpy as np
import pandas as pd
import yaml
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import LSTM, Dense, Input
from keras.models import Model
from keras.utils import plot_model
from tqdm import tqdm
from lib import utils_dalf

class TimeHistory(keras_callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


class DALFSupervisor():

    def __init__(self, is_training=True, **kwargs):
    
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._train_kwargs = kwargs.get('train')
        self._test_kwargs = kwargs.get('test')
        self._model_kwargs = kwargs.get('model')
        self._alg_name = self._kwargs.get('alg')

        # data args
        self._raw_dataset_dir = self._data_kwargs.get('raw_dataset_dir')
        self._test_size = self._data_kwargs.get('test_size')

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils_dalf.get_logger(self._log_dir, __name__, 'info.log', level=log_level)
        self._logger.info(kwargs)

        # Model's Args
        self._verified_percentage = self._model_kwargs.get('verified_percentage')
        self._rnn_units = self._model_kwargs.get('rnn_units')
        self._seq_len = self._model_kwargs.get('seq_len')
        self._num_hour = self._model_kwargs.get('num_hours')
        self._horizon = self._model_kwargs.get('horizon')
        self._input_dim = self._model_kwargs.get('input_dim')
        self._output_dim = self._model_kwargs.get('output_dim')
        self._nodes = self._model_kwargs.get('num_nodes')

        self._wh_mat = np.load(self._raw_dataset_dir)['weekend_holiday']

        # Train's args
        self._drop_out = self._train_kwargs.get('dropout')
        self._epochs = self._train_kwargs.get('epochs')
        self._batch_size = self._data_kwargs.get('batch_size')

        # Test's args
        self._run_times = self._test_kwargs.get('run_times')

        # Load data
        self._data = utils_dalf.load_dataset(seq_len=self._seq_len, horizon=self._horizon, num_hour = self._num_hour,
                                                input_dim=self._input_dim, output_dim=self._output_dim,
                                                raw_dataset_dir=self._raw_dataset_dir,
                                                verified_percentage=self._verified_percentage, p=self._test_size)

        self.callbacks_list = []

        self._checkpoints = ModelCheckpoint(
            self._log_dir + "best_model.hdf5",
            monitor='val_loss', verbose=1,
            save_best_only=True,
            mode='auto', period=1)
        self.callbacks_list = [self._checkpoints]

        self._earlystop = EarlyStopping(monitor='val_loss', patience=self._train_kwargs.get('patience'),
                                        verbose=1, mode='auto')
        self.callbacks_list.append(self._earlystop)

        self._time_callback = TimeHistory()
        self.callbacks_list.append(self._time_callback)

        self.model = self._model_construction(is_training=is_training)

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            seq_len = kwargs['model'].get('seq_len')
            horizon = kwargs['model'].get('horizon')
            input_dim = kwargs['model'].get('input_dim')
            verified_percentage = kwargs['model'].get('verified_percentage')

            run_id = '%d_%d_%d_%s_%d_%g/' % (
                seq_len, horizon, input_dim,
                structure, batch_size, verified_percentage)
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def _model_construction(self, is_training=True):
        # Model
        encoder_inputs = Input(shape=(None, self._input_dim))
        encoder = LSTM(self._rnn_units, return_state=True)
        _, state_h, state_c = encoder(encoder_inputs)
        # We discard `encoder_outputs` and only keep the states.
        encoder_states = [state_h, state_c]

        # Set up the decoder, using `encoder_states` as initial state.
        decoder_inputs = Input(shape=(None, self._output_dim))
        # We set up our decoder to return full output sequences,
        # and to return internal states as well. We don't use the
        # return states in the training model, but we will use them in inference.
        decoder_lstm = LSTM(self._rnn_units, return_sequences=True, return_state=True)
        decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                             initial_state=encoder_states)

        decoder_dense = Dense(self._output_dim, activation='relu')
        decoder_outputs = decoder_dense(decoder_outputs)

        # Define the model that will turn
        # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
        model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

        if is_training:
            return model
        else:
            self._logger.info("Load model from: {}".format(self._log_dir))
            model.load_weights(self._log_dir + 'best_model.hdf5')
            model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

            # Construct E_D model for predicting
            self.encoder_model = Model(encoder_inputs, encoder_states)

            decoder_state_input_h = Input(shape=(self._rnn_units,))
            decoder_state_input_c = Input(shape=(self._rnn_units,))
            decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
            decoder_outputs, state_h, state_c = decoder_lstm(
                decoder_inputs, initial_state=decoder_states_inputs)
            decoder_states = [state_h, state_c]
            decoder_outputs = decoder_dense(decoder_outputs)
            self.decoder_model = Model(
                [decoder_inputs] + decoder_states_inputs,
                [decoder_outputs] + decoder_states)

            plot_model(model=self.encoder_model, to_file=self._log_dir + '/encoder.png', show_shapes=True)
            plot_model(model=self.decoder_model, to_file=self._log_dir + '/decoder.png', show_shapes=True)

            return model

    def train(self):
        self.model.compile(optimizer='adam', loss='mse', metrics=['mse', 'mae'])

        training_history = self.model.fit([self._data['encoder_input_train'], self._data['decoder_input_train']],
                                          self._data['decoder_target_train'],
                                          batch_size=self._batch_size,
                                          epochs=self._epochs,
                                          callbacks=self.callbacks_list,
                                          validation_data=([self._data['encoder_input_val'],
                                                            self._data['decoder_input_val']],
                                                           self._data['decoder_target_val']),
                                          shuffle=True,
                                          verbose=2)
        if training_history is not None:
            self._plot_training_history(training_history)
            self._save_model_history(training_history)
            config = dict(self._kwargs)
            config_filename = 'config_lstm.yaml'
            config['train']['log_dir'] = self._log_dir
            with open(os.path.join(self._log_dir, config_filename), 'w') as f:
                yaml.dump(config, f, default_flow_style=False)

    def evaluate(self):
        # todo:
        pass

    def test(self):
        for time in range(self._run_times):
            print('TIME: ', time+1)
            self._test()

    def _test(self):
        scaler = self._data['scaler']
        data_test = self._data['test_data_norm']
        T = len(data_test)
        K = data_test.shape[1]
        wh_mat = np.expand_dims(self._wh_mat, axis = 1)
        wh_mat = np.repeat(wh_mat, K, axis = 1)
        
        bm = utils_dalf.binary_matrix(self._verified_percentage, len(data_test), self._nodes)
        l = self._seq_len
        h = self._horizon
        d = self._num_hour
        pd = np.zeros(shape=(T - h, self._nodes), dtype='float32')
        pd[:l*d] = data_test[:l*d]
        _pd = np.zeros(shape=(T - h, self._nodes), dtype='float32')
        _pd[:l*d] = data_test[:l*d]
        iterator = tqdm(range(0, T - l*d - d, d))
        for i in iterator:
            for k in range(K):
                input = np.zeros(shape=(d, l, self._input_dim))
                for ihour in range(d):
                    list_yhats = []
                    # input_dim = 3
                    input[ihour, :, 0] = pd[i:(i + l*d):d, k]
                    input[ihour, :, 1] = bm[i:(i + l*d):d, k]
                    input[ihour, :, 2] = wh_mat[i:(i+ l*d):d, k]
                    yhats = self._predict(input)
                    yhats = np.squeeze(yhats, axis=-1)
                    list_yhats.append(yhats)

                arr_yhats = np.asarray(list_yhats)
                _pd[i + (l*d):i + (l+1)*d, k] = arr_yhats
                # update y
                _bm = bm[i + l*d:i + (l+1)*d, k].copy()
                _gt = data_test[i + l*d:i + (l+1)*d, k].copy()
                pd[i + l*d:i + (l+1)*d, k] = arr_yhats * (1.0 - _bm) + _gt * _bm
        # save bm and pd to log dir
        np.savez(self._log_dir + "binary_matrix_and_pd", bm=bm, pd=pd)
        predicted_data = scaler.inverse_transform(_pd)
        ground_truth = scaler.inverse_transform(data_test[:_pd.shape[0]])
        np.save(self._log_dir+'pd', predicted_data)
        np.save(self._log_dir+'gt', ground_truth)
        # save metrics to log dir
        error_list = utils_dalf.cal_error(ground_truth.flatten(), predicted_data.flatten())
        utils_dalf.save_metrics(error_list, self._log_dir, self._alg_name)

    def _predict(self, source):
        states_value = self.encoder_model.predict(source)
        target_seq = np.zeros((1, 1, self._output_dim))

        yhat = np.zeros(shape=(self._horizon+1, 1),
                        dtype='float32')
        for i in range(self._horizon + 1):
            output_tokens, h, c = self.decoder_model.predict(
                [target_seq] + states_value)
            output_tokens = output_tokens[0, -1, 0]
            yhat[i] = output_tokens

            target_seq = np.zeros((1, 1, self._output_dim))
            target_seq[0, 0, 0] = output_tokens

            # Update states
            states_value = [h, c]
        return yhat[-self._horizon:]

    def load(self):
        self.model.load_weights(self._log_dir + 'best_model.hdf5')

    def _save_model_history(self, model_history):
        loss = np.array(model_history.history['loss'])
        val_loss = np.array(model_history.history['val_loss'])
        dump_model_history = pd.DataFrame(index=range(loss.size),
                                          columns=['epoch', 'loss', 'val_loss', 'train_time'])

        dump_model_history['epoch'] = range(loss.size)
        dump_model_history['loss'] = loss
        dump_model_history['val_loss'] = val_loss

        if self._time_callback.times is not None:
            dump_model_history['train_time'] = self._time_callback.times

        dump_model_history.to_csv(self._log_dir + 'training_history.csv', index=False)

    def _plot_training_history(self, model_history):
        import matplotlib.pyplot as plt

        plt.plot(model_history.history['loss'], label='loss')
        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

        plt.plot(model_history.history['val_loss'], label='val_loss')
        plt.savefig(self._log_dir + '[val_loss]{}.png'.format(self._alg_name))
        plt.legend()
        plt.close()

    def plot_models(self):
        plot_model(model=self.model, to_file=self._log_dir + '/model.png', show_shapes=True)


    def plot_series(self):
        from matplotlib import pyplot as plt
        preds = np.load(self._log_dir+'pd.npy')
        gt = np.load(self._log_dir+'gt.npy')

        for i in range(preds.shape[1]):
            plt.plot(preds[:, i], label='preds')
            plt.plot(gt[:, i], label='gt')
            plt.legend()
            plt.savefig(self._log_dir + '[result_predict]series_{}.png'.format(str(i+1)))
            plt.close()