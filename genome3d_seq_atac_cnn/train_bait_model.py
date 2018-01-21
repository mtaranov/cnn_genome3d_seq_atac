from __future__ import absolute_import

import argparse
import json
import logging
import numpy as np
import os
from pybedtools import BedTool
from sklearn.utils import shuffle

from genomedatalayer.extractors import (
    FastaExtractor, MemmappedBigwigExtractor, MemmappedFastaExtractor
)

from genome3d.intervals import get_tf_predictive_setup, train_test_chr_split 
from genome3d.models import SequencePairClassifier

# setup logging
log_formatter = \
    logging.Formatter('%(levelname)s:%(asctime)s:%(name)s] %(message)s')
logger = logging.getLogger('genome3d')
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
handler.setFormatter(log_formatter)
logger.setLevel(logging.INFO)
logger.addHandler(handler)
logger.propagate = False

RAW_INPUT_KEYS = ['dnase_bigwig', 'genome_fasta']
input2memmap_extractor = {'dnase_bigwig': MemmappedBigwigExtractor,
                          'genome_fasta': MemmappedFastaExtractor}
input2memmap_input = {'dnase_bigwig': 'dnase_data_dir',
                      'genome_fasta': 'genome_data_dir'}

logger.info("loading regions1, regions2, and labels...")
train_args = {"regions1_train": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site1_train_dist_matched_thres10.bed"),
              "regions2_train": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site2_train_dist_matched_thres10.bed"),
              "regions1_test": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site1_test_dist_matched_thres10.bed"),
              "regions2_test": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site2_test_dist_matched_thres10.bed"),
#train_args = {"regions1_train": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site1_train_dist_matched_10kb_thres10.bed"),
#              "regions2_train": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site2_train_dist_matched_10kb_thres10.bed"),
#              "regions1_test": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site1_test_dist_matched_10kb_thres10.bed"),
#              "regions2_test": BedTool("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/site2_test_dist_matched_10kb_thres10.bed"),
              "labels_train": np.load("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/labels_train_dist_matched_thres10.npy"),
              "labels_test": np.load("/users/mtaranov/NN_thres5max_datasets/dist_matched/regions/labels_test_dist_matched_thres10.npy"),
              #"genome_data_dir": "/mnt/data/memmap/GGR/ATACSEQ/primary_keratinocyte-d00.GGR.Stanford_Greenleaf.ATAC-seq.b1.trim.PE2SE.nodup.tn5_pooled.pf.fc.signal.bigwig",
              #"dnase_data_dir": "/srv/scratch/mtaranov/atac_pval_D0",
              "dnase_data_dir": "/srv/scratch/mtaranov/atac_fc_D0",
              "genome_data_dir": "/mnt/data/memmap/genomes/hg19.GRCh37/",
              "prefix": "test_run"}

def main_train(regions1_train=None,
               regions2_train=None,
               regions1_test=None,
               regions2_test=None,
               labels_train=None,
               labels_test=None,
               genome_data_dir=None,
               dnase_data_dir=None,
               prefix=None,
               n_jobs=None,
               arch_file=None,
               weights_file=None):
    #labels = labels.astype(int)[:1000,:]
    regions1_train = list(regions1_train)
    regions2_train = list(regions2_train)
    regions1_test = list(regions1_test)
    regions2_test = list(regions2_test)
    y_train = labels_train.astype(int)
    y_test = labels_test.astype(int)

    logger.info("splitting data into training and testing..")
    #regions1_train, regions1_test, regions2_train, regions2_test, y_train, y_test = train_test_chr_split(regions1, regions2, labels, ["chr16", "chrX"])
    #regions1_train, regions1_test, regions2_train, regions2_test, y_train, y_test = get_train_test_dist_matched("/users/mtaranov/3D_fromATAC_scg4/data/all_atac_1kb_contacts_w_labels.bed")
    logger.info("Number of examples for testing: {}".format(len(y_test)))
    logger.info("Shuffling training data..")
    regions1_train, regions2_train, y_train = shuffle(regions1_train, regions2_train, y_train, random_state=0)
    """
    logger.info("Subsamping negative training data..")
    pos_indxs = np.where(y_train == 1)[0]
    neg_indxs = np.where(y_train == 0)[0]
    subsampled_indxs = np.concatenate((pos_indxs, neg_indxs[:len(pos_indxs)]))
    subsampled_indxs = shuffle(subsampled_indxs, random_state=0)
    regions1_train = [regions1_train[indx] for indx in subsampled_indxs]
    regions2_train = [regions2_train[indx] for indx in subsampled_indxs]
    y_train = y_train[subsampled_indxs]
    """
    # set up the architecture
    interval_length = regions1_train[0].length
    num_tasks = 1
    architecture_parameters = {'num_filters': (30, 30),
                               'conv_width': (10, 10),
                               'num_combined_filters': (30, 30),
                               'combined_conv_width': (5, 5),
                               'dropout': 0.0}
#    architecture_parameters = {'num_filters': (50, 50),
#                               'conv_width': (25, 25),
#                               'num_combined_filters': (50,),
#                               'combined_conv_width': (25,),
#                               'pool_width': 35,
#                               'dropout': 0.0}
#    architecture_parameters = {'num_filters': (50, 50),
#                               'conv_width': (25, 25),
#                               'num_combined_filters': (),
#                               'combined_conv_width': (),
#                               'pool_width': 20,
#                               'use_RNN': True,
#                               'GRU_size': 20,
#                               #'TDD_size': 15,
#                               'dropout': 0.0}
    fasta_extractor = MemmappedFastaExtractor(genome_data_dir)
    #dnase_extractor = MemmappedBigwigExtractor(dnase_data_dir, local_norm_half_width=interval_length/2)
    dnase_extractor = MemmappedBigwigExtractor(dnase_data_dir)
    #fasta_extractor = MemmappedBigwigExtractor(genome_data_dir)
    logger.info("Found memmapped fastas, initialized memmaped fasta extractors")
    # get appropriate model class
    model_class = SequencePairClassifier
    # Initialize, compile, and train
    logger.info("Initializing a {}".format(model_class))
    model = model_class(interval_length, num_tasks, **architecture_parameters)
    logger.info("Compiling {}..".format(model_class))
    model.compile(y=y_train)
    logger.info("Starting to train with streaming data..")
    #model.train((regions1_train, regions2_train, y_train), (regions1_test, regions2_test, y_test), [dnase_extractor], batch_size=128, epoch_size=10000)
    #model.train((regions1_train, regions2_train, y_train), (regions1_test, regions2_test, y_test), [fasta_extractor], epoch_size=15000)
    #model.train((regions1_train, regions2_train, y_train), (regions1_test, regions2_test, y_test), [dnase_extractor], epoch_size=15000)
    #model.train((regions1_train, regions2_train, y_train), (regions1_test, regions2_test, y_test), [fasta_extractor], epoch_size=28670)
    model.train((regions1_train, regions2_train, y_train), (regions1_test, regions2_test, y_test), [dnase_extractor], epoch_size=28670)
    #model.train((regions1_train, regions2_train, y_train), (regions1_test, regions2_test, y_test), [fasta_extractor],
    #            epoch_size=28670)
    #logger.info("Saved trained model files to {}.arch.json and {}.weights.h5".format(prefix, prefix))
    logger.info("Done!")

if __name__ == "__main__":
    logger.info("running main_train...")
    main_train(**train_args)
