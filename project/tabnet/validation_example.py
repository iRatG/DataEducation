import numpy as np
import tensorflow as tf

from tqdm import tqdm

class Validation:
    def __init__(self, cfg, data_loader, valid_writer, network, dataset_type):
        self.cfg = cfg
        self.data_loader = data_loader
        self.valid_summary_writer = valid_writer
        self.network = network
        self.dataset_type = dataset_type

        self.__create_metrics()

    def __create_metrics(self):
        self.log_loss = tf.losses.BinaryCrossentropy()
        self.loss_metric = tf.keras.metrics.Mean(name='%s_loss' % self.dataset_type)
        self.AUC_metric = tf.keras.metrics.AUC(name='%s_AUC' % self.dataset_type)
        self.binary_metric = tf.keras.metrics.BinaryAccuracy(name='%s_BinaryAccuracy' % self.dataset_type)

        return

    def __reset_all_metrics(self):
        self.loss_metric.reset_states()
        self.AUC_metric.reset_states()
        self.binary_metric.reset_states()

        return

    def run_validation(self, step):
        print('Run Validation for %d step' % step)
        self.__reset_all_metrics()
        
        last_step_aggregated_mask_values_all = []
        for features, labels in tqdm(self.data_loader.oot_loader if self.dataset_type == 'OOT' else self.data_loader.valid_loader):
            output, output_aggregated, total_entropy, \
            aggregated_mask_values_all, mask_values_all = self.network(features, training=False)

            loss = self.log_loss(labels, output)

            self.loss_metric(loss)
            self.AUC_metric(labels, output)
            self.binary_metric(labels, output)
            
            last_step_aggregated_mask_values_all.extend(aggregated_mask_values_all[-1])
        
        sorted_featrues_by_import = tf.reduce_mean(last_step_aggregated_mask_values_all, axis=0).numpy().argsort()[::-1]
        sorted_features_name = [self.data_loader.get_columns()[idx].name for idx in sorted_featrues_by_import]
            
        with self.valid_summary_writer.as_default():
            tf.summary.scalar('Log_Loss', self.loss_metric.result(), step=step)
            tf.summary.scalar('AUC_score', self.AUC_metric.result(), step=step)
            tf.summary.scalar('Accuracy', self.binary_metric.result(), step=step)
            tf.summary.scalar('Gini_index', 2 * self.AUC_metric.result() - 1, step=step)
            tf.summary.text('Features_by_importance', tf.stack(sorted_features_name), step=step)

        template = '{} for Step {}, loss: {}, binary accuracy: {}, AUC: {}'
        print(template.format(self.dataset_type, step, self.loss_metric.result(), self.binary_metric.result() * 100,
                              self.AUC_metric.result() * 100))

        return
