from __future__ import absolute_import, division, print_function

from builtins import zip
import itertools
import numpy as np
import os
import psutil

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import (
    Activation, Dense, Dropout, Flatten, Merge,
    Permute, Reshape, TimeDistributedDense
)
from keras.layers.convolutional import Convolution2D, MaxPooling2D, AveragePooling2D
from keras.layers.recurrent import GRU
from keras import optimizers
from keras.regularizers import l1
from keras.utils.generic_utils import Progbar

from .io_utils import generate_from_interval_pairs, generate_from_interval_pairs_and_labels, roundrobin
from .metrics import ClassificationResult, AMBIG_LABEL

def class_weights(y):
    """
    Parameters
    ----------
    y : 1darray
    """
    assert len(np.shape(y))==1
    total = (y >= 0).sum()
    num_neg = (y == 0).sum()
    num_pos = (y == 1).sum()

    return 0.1 * total / num_neg, 5 * total / num_pos


def get_weighted_binary_crossentropy(w0_weights, w1_weights):
    # Compute the task-weighted cross-entropy loss, where every task is weighted by 1 - (fraction of non-ambiguous examples that are positive)
    # In addition, weight everything with label -1 to 0
    w0_weights = np.array(w0_weights)
    w1_weights = np.array(w1_weights)
    def binary_crossentropy(y_true, y_pred): 
        weightsPerTaskRep = y_true * w1_weights[None, :] + (1 - y_true) * w0_weights[None, :]
        nonAmbig = (y_true > 0.5)
        nonAmbigTimesWeightsPerTask = nonAmbig * weightsPerTaskRep
        return K.mean(K.binary_crossentropy(y_pred, y_true) * nonAmbigTimesWeightsPerTask, axis=-1)
    return binary_crossentropy


class SequencePairClassifier(object):
    def __init__(self, seq_length=None, num_tasks=None, arch_fname=None, weights_fname=None,
                 num_filters=(15, 15, 15), conv_width=(15, 15, 15),
                 num_combined_filters=(15, 15, 15), combined_conv_width=(15, 15, 15),
                 pool_width_first_lyr=20, pool_width=5,  L1=0, dropout=0.0,
                 use_RNN=False, GRU_size=35, TDD_size=15,
                 num_epochs=100, verbose=2):
        self.saved_params = locals()
        self.verbose = verbose
        self.num_epochs = num_epochs
        if arch_fname is not None and weights_fname is not None:
            from keras.models import model_from_json
            self.model = model_from_json(open(arch_fname).read())
            self.model.load_weights(weights_fname)
            self.num_tasks = self.model.layers[-1].output_shape[-1]
        elif seq_length is not None and num_tasks is not None:
            self.num_tasks = num_tasks
            self.model = Sequential()
            assert len(num_filters) == len(conv_width)
            for i, (nb_filter, nb_col) in enumerate(zip(num_filters, conv_width)):
                # for seq 
                conv_height = 4 if i == 0 else 1
                # for ATAC-seq
                #conv_height = 1
                #if i == 0:
                #    self.model.add(MaxPooling2D(pool_size=(conv_height, pool_width_first_lyr), input_shape=(1, conv_height*2, seq_length)))
                    #self.model.add(AveragePooling2D(pool_size=(conv_height, pool_width_first_lyr), input_shape=(1, conv_height*2, seq_length)))
                self.model.add(Convolution2D(
                    nb_filter=nb_filter, nb_row=conv_height,
                    nb_col=nb_col, activation='linear', subsample=(conv_height, 1),
                    init='he_normal', input_shape=(1, conv_height*2, seq_length),
                    #init='he_normal', 
                    W_regularizer=l1(L1), b_regularizer=l1(L1)))
#                self.model.add(Activation('relu'))
#                self.model.add(Dropout(dropout))
            self.model.add(MaxPooling2D(pool_size=(1, pool_width)))
            conv_height = 2
            for i, (nb_filter, nb_col) in enumerate(zip(num_combined_filters, combined_conv_width)):
                conv_height = 2 if i == 0 else 1
                self.model.add(Convolution2D(
                    nb_filter=nb_filter, nb_row=conv_height,
                    nb_col=nb_col, activation='linear',
                    init='he_normal', input_shape=(1, conv_height, seq_length),
                    W_regularizer=l1(L1), b_regularizer=l1(L1)))
                self.model.add(Activation('relu'))
                self.model.add(Dropout(dropout))
            #self.model.add(MaxPooling2D(pool_size=(1, 1)))
            if use_RNN:
                num_max_pool_outputs = self.model.layers[-1].output_shape[-1]
                if len(num_combined_filters) > 0:
                    self.model.add(Reshape((2 * num_combined_filters[-1], num_max_pool_outputs)))
                else:
                    self.model.add(Reshape((2 * num_filters[-1], num_max_pool_outputs)))
                self.model.add(Permute((2, 1)))
                self.model.add(GRU(GRU_size, return_sequences=True))
                #self.model.add(TimeDistributedDense(TDD_size, activation='linear'))
            #self.model.add(AveragePooling2D(pool_size=(1, pool_width)))
            self.model.add(MaxPooling2D(pool_size=(1, pool_width)))

            self.model.add(Flatten())
            self.model.add(Dense(300))
            self.model.add(Activation('relu'))
            self.model.add(Dense(300))
            self.model.add(Activation('relu'))
            self.model.add(Dense(output_dim=self.num_tasks))
            self.model.add(Activation('sigmoid'))
        else:
            raise RuntimeError("Model initialization requires seq_length and num_tasks or arch/weights files!")

    #def compile(self, optimizer='adam', lr=0.0001, y=None):
    #def compile(self, optimizer='sgd', lr=0.000001, y=None):
    def compile(self, optimizer='sgd', lr=0.000001, y=None):
        """
        Defines learning parameters and compiles the model.

        Parameters
        ----------
        y : 2darray, optional
           Uses class-weighted cross entropy loss if provides.
           Otherwise, uses non-weighted cross entropy loss.
        """
        if y is not None:
            task_weights = np.array([class_weights(y[:, i]) for i in range(y.shape[1])])
            loss_func = get_weighted_binary_crossentropy(task_weights[:, 0], task_weights[:, 1])
        else:
            loss_func='binary_crossentropy'
        optimizer_cls = getattr(optimizers, optimizer)
        optimizer = optimizer_cls(lr=lr)
        self.model.compile(optimizer=optimizer, loss=loss_func)

    def train(self, train_region_pairs_and_labels, valid_region_pairs_and_labels, fasta_extractor,
              task_names=None, save_best_model_to_prefix=None,
              early_stopping_metric='auROC', num_epochs=100,
              batch_size=128, epoch_size=250000,
              early_stopping_patience=5, verbose=True):
        # create dictionaries so we could call train_on_multiple_datasets
        dataset2train_region_pairs_and_labels = {"only_dataset": train_region_pairs_and_labels}
        dataset2valid_region_pairs_and_labels = {"only_dataset": valid_region_pairs_and_labels}
        dataset2fasta_extractor = {"only_dataset": fasta_extractor}
        # call train_on_multiple_datasets
        self.train_on_multiple_datasets(dataset2train_region_pairs_and_labels, dataset2valid_region_pairs_and_labels, dataset2fasta_extractor,
                                        task_names=task_names, save_best_model_to_prefix=save_best_model_to_prefix,
                                        early_stopping_metric=early_stopping_metric, num_epochs=num_epochs,
                                        batch_size=batch_size, epoch_size=epoch_size,
                                        early_stopping_patience=early_stopping_patience, verbose=verbose)
        

    def train_on_multiple_datasets(self, dataset2train_region_pairs_and_labels, dataset2valid_region_pairs_and_labels, dataset2fasta_extractor,
                                   task_names=None, save_best_model_to_prefix=None,
                                   early_stopping_metric='auROC', num_epochs=100,
                                   batch_size=500, epoch_size=250000,
                                   early_stopping_patience=5, verbose=True):
        process = psutil.Process(os.getpid())
        # define training generator
        dataset2training_generator = {}
        for dataset_id, (regions1, regions2, labels) in dataset2train_region_pairs_and_labels.items():
           dataset2training_generator[dataset_id] = generate_from_interval_pairs_and_labels(regions1, regions2, labels, dataset2fasta_extractor[dataset_id],
                                                                                            batch_size=batch_size, indefinitely=True)
        training_generator = roundrobin(*dataset2training_generator.values())
        # define training loop
        valid_metrics = []
        best_metric = np.inf if early_stopping_metric == 'Loss' else -np.inf
        samples_per_epoch = len(y_train) if epoch_size is None else epoch_size
        batches_per_epoch = int(samples_per_epoch / batch_size)
        samples_per_epoch = batch_size * batches_per_epoch
        for epoch in range(1, num_epochs + 1):
            progbar = Progbar(target=samples_per_epoch)
            for batch_indxs in xrange(1, batches_per_epoch + 1):
                x, y = next(training_generator)
                print(x[0].shape)
                print(x[0].dtype)
                batch_loss = self.model.train_on_batch(x, y)
                rss_minus_shr_memory = (process.memory_info().rss -  process.memory_info().shared)  / 10**6
                progbar.update(batch_indxs*batch_size,
                               values=[("loss", sum(batch_loss)/len(batch_loss)), ("Non-shared RSS (Mb)", rss_minus_shr_memory)])

            #dataset2metrics, epoch_valid_metrics = self.test_on_multiple_datasets(dataset2valid_region_pairs_and_labels, dataset2fasta_extractor,
            #                                                                      task_names=task_names)
            dataset2metrics, epoch_valid_metrics = self.test_on_multiple_datasets(dataset2valid_region_pairs_and_labels, dataset2fasta_extractor,
                                                                                  task_names=task_names, batch_size=batch_size)
            #maryna 
            dataset2metrics_train, epoch_train_metrics = self.test_on_multiple_datasets(dataset2train_region_pairs_and_labels, dataset2fasta_extractor,
                                                                                  task_names=task_names, batch_size=batch_size)
            # maryna print train_val 
            #dataset2metrics, epoch_train_metrics = self.test_on_multiple_datasets(dataset2train_region_pairs_and_labels, dataset2fasta_extractor,
            #                                                                      task_names=task_names, batch_size=batch_size)
            valid_metrics.append(epoch_valid_metrics)
            if verbose:
                print('\nEpoch {}:'.format(epoch))
                for dataset_id, dataset_metrics in dataset2metrics.items():
                    print('Valid Dataset {}:\n{}\n'.format(dataset_id, dataset_metrics), end='')
                #maryna
                for dataset_id_train, dataset_metrics_train in dataset2metrics_train.items():
                    print('Train Dataset {}:\n{}\n'.format(dataset_id_train, dataset_metrics_train), end='')
                if len(dataset2metrics) > 1:
                    print('Metrics across all datasets:\n{}\n'.format(epoch_valid_metrics), end='')
            current_metric = epoch_valid_metrics[early_stopping_metric].mean()
            if (early_stopping_metric == 'Loss') == (current_metric <= best_metric):
                if verbose:
                    print('New best {}. Saving model.\n'.format(early_stopping_metric))
                best_metric = current_metric
                best_epoch = epoch
                early_stopping_wait = 0
                if save_best_model_to_prefix is not None:
                    self.save(save_best_model_to_prefix)
            else:
                if early_stopping_wait >= early_stopping_patience:
                    break
                early_stopping_wait += 1
        if verbose: # end of training messages
            print('Finished training after {} epochs.'.format(epoch))
            if save_best_model_to_prefix is not None:
                print("The best model's architecture and weights (from epoch {0}) "
                      'were saved to {1}.arch.json and {1}.weights.h5'.format(
                    best_epoch, save_best_model_to_prefix))

    def predict(self, intervals1, intervals2, extractors, batch_size=128, verbose=True):
        """
        Generates data and returns a single 2d array with predictions.
        """
        generator = generate_from_interval_pairs(intervals1, intervals2, extractors,
                                                 batch_size=batch_size, indefinitely=False)
        if verbose:
            predictions = []
            num_samples = len(intervals1)
            progbar = Progbar(target=num_samples)
            for batch_indx, batch in enumerate(generator):
                predictions.append(np.vstack(self.model.predict_on_batch(batch)))
                if batch_indx*batch_size <= num_samples:
                    progbar.update(batch_indx*batch_size)
                else:
                    progbar.update(num_samples)
        else:
            predictions = [np.vstack(self.model.predict_on_batch(batch))
                           for batch in generator]
        return np.vstack(predictions)

    def test(self, intervals1, intervals2, y, extractors, task_names=None):
        predictions = self.predict(intervals1, intervals2, extractors)
        return ClassificationResult(y, predictions, task_names=task_names)

    def test_on_multiple_datasets(self, dataset2test_region_pairs_and_labels, dataset2fasta_extractor,
                                  batch_size=128, task_names=None):
        """
        Returns dctionary with dataset ids as keys and classification results as values, and a combined classification result.
        """
        multiple_datasets = len(dataset2test_region_pairs_and_labels) > 1
        dataset2classification_result = {}
        predictions_list = []
        labels_list = []
        for dataset_id, (regions1, regions2, labels) in dataset2test_region_pairs_and_labels.items():
            predictions = self.predict(regions1, regions2, dataset2fasta_extractor[dataset_id], batch_size=batch_size)
            dataset2classification_result[dataset_id] = ClassificationResult(labels, predictions, task_names=task_names)
            if multiple_datasets:
                predictions_list.append(predictions)
                labels_list.append(labels)
        if multiple_datasets:
            predictions = np.vstack(predictions_list)
            y = np.vstack(labels_list)
            combined_classification_result = ClassificationResult(y, predictions, task_names=task_names)
        else:
            combined_classification_result = dataset2classification_result.values()[0]

        return (dataset2classification_result, combined_classification_result)

    def save(self, prefix):
        arch_fname = prefix + '.arch.json'
        weights_fname = prefix + '.weights.h5'
        open(arch_fname, 'w').write(self.model.to_json())
        self.model.save_weights(weights_fname, overwrite=True)

    def score(self, X, y, metric):
        return self.test(X,y)[metric]
