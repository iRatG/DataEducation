import numpy as np
import os
import pickle
import tensorflow as tf
import pandas as pd
import gc

from tqdm import tqdm
from datetime import datetime

class DataLoader:
    def __init__(self, cfg):
        self.cfg = cfg
        
        self.DEFAULTS = (
                    [[0] for col in self.cfg.dataset.INT_COLUMNS] + [[""] for col in self.cfg.dataset.BOOL_COLUMNS] +
                    [[0.0] for col in self.cfg.dataset.FLOAT_COLUMNS] + [[""] for col in
                                                                         self.cfg.dataset.COLUMNS_CATEGORY] +
                    [[-1]])

        self.FEATURE_COLUMNS = (
                self.cfg.dataset.INT_COLUMNS + self.cfg.dataset.BOOL_COLUMNS +
                self.cfg.dataset.COLUMNS_CATEGORY + self.cfg.dataset.FLOAT_COLUMNS +
                self.cfg.dataset.STR_COLUMNS_BUCKET + self.cfg.dataset.STR_COLUMNS_HASH_BUCKET
        )
        self.ALL_COLUMNS = self.FEATURE_COLUMNS + [self.cfg.dataset.LABEL_COLUMN]

        self.num_features = len(self.cfg.dataset.INT_COLUMNS) + len(self.cfg.dataset.BOOL_COLUMNS) * self.cfg.train_bool_feat_dim + \
                            len(self.cfg.dataset.COLUMNS_CATEGORY) * self.cfg.train_category_feat_dim + \
                            len(self.cfg.dataset.FLOAT_COLUMNS) + len(self.cfg.dataset.STR_COLUMNS_BUCKET) +                    len(self.cfg.dataset.STR_COLUMNS_HASH_BUCKET)

        self.__read_data()
        self.__create_loaders()

        return

    def __read_data(self):
        print('Train Data loading ...')
        with open(self.cfg.dataset.train_dir_path, 'rb') as f:
            train_data = pickle.load(f)['df']

        self.train_Y = train_data['TARGET'].values
        
        self.train_extra_columns = train_data[self.cfg.dataset.extra_columns].copy()
        
        self.train_X = train_data.drop(self.cfg.dataset.removed_columns, axis=1).copy()
        self.train_X = self.train_X[self.FEATURE_COLUMNS].copy()
        
        del train_data

        for col in self.cfg.dataset.COLUMNS_CATEGORY:
            print(col, self.train_X[col].unique())

        self.negative_indices = np.where(self.train_Y == 0)[0]
        self.positive_indices = np.where(self.train_Y == 1)[0]
        self.nrof_batches = len(self.train_X) // self.cfg.train.batch_size
        
        print('Valid Data loading ...')
        with open(self.cfg.dataset.valid_dir_path, 'rb') as f:
            valid_data = pickle.load(f)['df']

        self.valid_Y = valid_data['TARGET'].values
        self.valid_extra_columns = valid_data[self.cfg.dataset.extra_columns]
        self.valid_X = valid_data.drop(self.cfg.dataset.removed_columns, axis=1)
        self.valid_X = self.valid_X[self.FEATURE_COLUMNS]
        
        del valid_data
        
        gc.collect()

        return
    
    def __create_loaders(self):
        print('Create data loaders ...')
        self.train_loader = tf.data.Dataset.from_tensor_slices((dict(self.train_X), self.train_Y))
        self.train_loader = self.train_loader.batch(self.cfg.train.batch_size).prefetch(buffer_size=self.cfg.train.batch_size)

        self.valid_loader = tf.data.Dataset.from_tensor_slices((dict(self.valid_X), self.valid_Y))
        self.valid_loader = self.valid_loader.batch(self.cfg.validation.batch_size)

        return
    
    def get_columns(self):
        """Get the representations for all input columns."""
        columns = []
        if self.cfg.dataset.FLOAT_COLUMNS:
            columns += [tf.feature_column.numeric_column(ci, default_value=0.0) for ci in self.cfg.dataset.FLOAT_COLUMNS]
        if self.cfg.dataset.INT_COLUMNS:
            columns += [tf.feature_column.numeric_column(ci, default_value=0) for ci in self.cfg.dataset.INT_COLUMNS]
        # if self.STR_COLUMNS:
        #     # pylint: disable=g-complex-comprehension
        #     columns += [
        #         tf.feature_column.embedding_column(
        #             tf.feature_column.categorical_column_with_hash_bucket(
        #                 ci, hash_bucket_size=int(3 * num)),
        #             dimension=1) for ci, num in zip(self.STR_COLUMNS, self.STR_NUNIQUESS)
        #     ]
        if self.cfg.dataset.COLUMNS_CATEGORY:
            columns += [
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_vocabulary_list(key=ci,
                                                                              vocabulary_list=values),
                    dimension=self.cfg.train_category_feat_dim) for ci, values in zip(self.cfg.dataset.COLUMNS_CATEGORY,
                                                                                      self.cfg.dataset.COLUMNS_CATEGORY_NUNIQUESS)
            ]
        if self.cfg.dataset.BOOL_COLUMNS:
            columns += [
                tf.feature_column.embedding_column(
                    tf.feature_column.categorical_column_with_hash_bucket(
                        ci, hash_bucket_size=3),
                    dimension=self.cfg.train_bool_feat_dim) for ci in self.cfg.dataset.BOOL_COLUMNS
            ]
        return columns

    def shuffle_data(self):
        print('Shuffle data...')
        self.__create_loaders()

        return
    