import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_DUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
import numpy as np
import h5py
import pickle
import pandas as pd

from tqdm import tqdm
from tensorboard.plugins.hparams import api as hp
from datetime import datetime

from data_loader_example import DataLoader
from config_example import cfg
from network import TabNet
from validation import Validation
from plots import Plots

print(tf.test.is_gpu_available())

class Inflow:
    
    def __init__(self):
        self.data_loader = DataLoader(cfg)
        self.__build_model()

        self.log_config('config', cfg, self.train_summary_writer, 0)

        self.validation = Validation(cfg.validation, self.data_loader, self.valid_summary_writer, self.network, 'Valid')

    def log_config(self, name, config, summary_writer, step):
        general_keys = list(config.keys())
        rows = []
        for key in general_keys:
            try:
                subkeys = list(config[key])
                for subkey in subkeys:
                    rows.append(['%s.%s' % (key, subkey), str(config[key][subkey])])
            except:
                rows.append([key, str(config[key])])

        hyperparameters = [tf.convert_to_tensor(row) for row in rows]
        with summary_writer.as_default():
            tf.summary.text(name, tf.stack(hyperparameters), step=step)

        return

    def __build_model(self):
        self.network = TabNet(self.data_loader.get_columns(),
                             self.data_loader.num_features,
                             feature_dim=cfg.train.feature_dim,
                             output_dim=cfg.train.output_dim,
                             num_decision_steps=cfg.train.num_decision_steps,
                             relaxation_factor=cfg.train.relaxation_factor,
                             batch_momentum=cfg.train.batch_momentum,
                             virtual_batch_size=cfg.train.virtual_batch_size,
                             num_classes=cfg.train.num_classes,
                             encoder_type=cfg.train.encoder_type,
                             epsilon=0.00001)

        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            cfg.train.learning_rate,
            decay_steps=cfg.train.learning_rate_decay_steps,
            decay_rate=cfg.train.learning_rate_decay_factor,
            staircase=True
        )
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        self.log_loss = tf.losses.BinaryCrossentropy()
        self.loss_metric = tf.keras.metrics.Mean(name='train_loss')
        self.AUC_metric = tf.keras.metrics.AUC(name='train_AUC')
        self.binary_metric = tf.keras.metrics.BinaryAccuracy(name='train_BinaryAccuracy')

        train_log_dir = os.path.join(cfg.train.logs_base_dir, 'train')
        valid_log_dir = os.path.join(cfg.train.logs_base_dir, 'valid')
        self.train_summary_writer = tf.summary.create_file_writer(train_log_dir)
        self.valid_summary_writer = tf.summary.create_file_writer(valid_log_dir)
        return

    def __reset_all_metrics(self):
        self.loss_metric.reset_states()
        self.AUC_metric.reset_states()
        self.binary_metric.reset_states()

        return

    def __load_variables(self):
        if cfg.train.restore_model_path != '':
            print(cfg.train.restore_model_path)
            output = self.network(dict(self.data_loader.valid_X.take([0, 1], axis=0)), training=False)
            
            file = h5py.File(cfg.train.restore_model_path, 'r')
            weights = []
            for i in range(len(file.keys())):
                weights.append(file['weight' + str(i)].value)
            self.network.set_weights(weights)
            file.close()

#             file = h5py.File(cfg.train.restore_model_path.replace('model-', 'optimizer-'), 'r')
#             weights = []
#             for i in range(len(file.keys())):
#                 weights.append(file['weight' + str(i)].value)
#             self.optimizer.set_weights(weights)
#             file.close()

            step = int(cfg.train.restore_model_path.split('.h5')[0].split('-')[-1])
            self.optimizer.iterations.assign(step)

            print('Model restored')
        else:
            step = 0

        return step

    def __save_model(self, step):
        if not os.path.exists(cfg.train.models_base_dir):
            os.makedirs(cfg.train.models_base_dir)

        file_path = os.path.join(cfg.train.models_base_dir, 'model-%d.h5' % step)
        file = h5py.File(file_path, 'w')
        weights = self.network.get_weights()
        for i in range(len(weights)):
            file.create_dataset('weight' + str(i), data=weights[i])
        file.close()

        file_path = os.path.join(cfg.train.models_base_dir, 'optimizer-%d.h5' % step)
        file = h5py.File(file_path, 'w')
        weights = self.optimizer.get_weights()
        for i in range(len(weights)):
            file.create_dataset('weight' + str(i), data=weights[i])
        file.close()

        print('Model saved')

        return

    def run_train(self):
        step = self.__load_variables()

        self.validation.run_validation(step)

        for epoch in range(cfg.train.max_nrof_epochs):
            print('Start epoch %d' % epoch)

            self.__reset_all_metrics()

            batch_id = 0
            for batch_features, labels in tqdm(self.data_loader.train_loader):
                with tf.GradientTape() as tape:
                    output, output_aggregated, total_entropy,\
                    aggregated_mask_values_all, mask_values_all = self.network(batch_features, training=True)

                    loss = self.log_loss(labels, output)
                    # reg_loss = sum(self.network.losses)
                    reg_loss = cfg.train.weight_decay * tf.add_n([tf.nn.l2_loss(w) for w in self.network.trainable_variables])
                    total_loss = loss + cfg.train.sparsity_loss_weight * total_entropy + reg_loss

                grads = tape.gradient(total_loss, self.network.trainable_variables)
                capped_gvs = [tf.clip_by_value(grad, -cfg.train.gradient_thresh,
                                                cfg.train.gradient_thresh) for grad in grads]

                self.optimizer.apply_gradients(zip(capped_gvs, self.network.trainable_variables))

                self.loss_metric(total_loss)
                self.AUC_metric(labels, output)
                self.binary_metric(labels, output)

                with self.train_summary_writer.as_default():
                    # Visualization of the feature selection mask at decision step ni
                    for ni in range(len(mask_values_all)):
                        tf.summary.image(
                            "Mask for step" + str(ni),
                            tf.expand_dims(tf.expand_dims(mask_values_all[ni], 0), 3),
                            max_outputs=1, step=step)
                    # Visualization of the aggregated feature importances
                    for ni in range(len(aggregated_mask_values_all)):
                        tf.summary.image(
                            "Aggregated mask",
                            tf.expand_dims(tf.expand_dims(aggregated_mask_values_all[ni], 0), 3),
                            max_outputs=1, step=step)

                    tf.summary.scalar('Total_loss', self.loss_metric.result(), step=step)
                    tf.summary.scalar('Total entropy', total_entropy, step=step)
                    tf.summary.scalar('Log_Loss', loss, step=step)
                    tf.summary.scalar('Reg_Loss', reg_loss, step=step)
                    tf.summary.scalar('AUC_score', self.AUC_metric.result(), step=step)
                    tf.summary.scalar('Accuracy', self.binary_metric.result(), step=step)
                    tf.summary.scalar('Gini_index', 2 * self.AUC_metric.result() - 1, step=step)
                    tf.summary.scalar('Learning_rate', self.optimizer.learning_rate.__call__(step).numpy(), step=step)

                template = 'Epoch [{}][{}/{}], loss: {}, binary accuracy: {}, AUC: {}'
                print(template.format(epoch, batch_id, self.data_loader.nrof_batches,
                                      loss, self.binary_metric.result() * 100,
                                      self.AUC_metric.result() * 100))

                batch_id += 1
                step += 1
            
            self.__save_model(step)
            
            self.validation.run_validation(step)
            
            self.data_loader.shuffle_data()
        
        return
    