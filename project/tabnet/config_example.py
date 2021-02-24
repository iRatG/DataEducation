import numpy as np

from datetime import datetime
from easydict import EasyDict

cfg = EasyDict()

cfg.dataset = EasyDict()
cfg.dataset.train_dir_path = ""
cfg.dataset.valid_dir_path = ""

cfg.dataset.removed_columns = []
cfg.dataset.extra_columns = []

cfg.dataset.TO_STR = []
cfg.dataset.TO_INT = []
cfg.dataset.BOOL_COLUMNS = []
cfg.dataset.INT_COLUMNS = [] # 'MOB'
cfg.dataset.COLUMNS_CATEGORY = []
cfg.dataset.COLUMNS_CATEGORY_NUNIQUESS = []
cfg.dataset.STR_COLUMNS_BUCKET = []
cfg.dataset.STR_COLUMNS_HASH_BUCKET = []
cfg.dataset.LABEL_COLUMN = "TARGET"

cfg.dataset.FLOAT_COLUMNS = []

cfg.train = EasyDict()

cfg.train.run_model_path = './run_TabNet_%s' % (datetime.now().strftime("%Y-%m-%d-%H%M%S"))
cfg.train.logs_base_dir = cfg.train.run_model_path + '/logs'
cfg.train.models_base_dir = cfg.train.run_model_path + '/models'
cfg.train.restore_model_path = ""

cfg.train.weight_decay = 0.0005
cfg.train.learning_rate = 0.02
cfg.train.learning_rate_decay_steps = 150
cfg.train.learning_rate_decay_factor = 0.99
cfg.train.batch_size = 16384
cfg.train.prefetch_batch_size = 1024
cfg.train.max_nrof_epochs = 100000
cfg.train.sparsity_loss_weight = 0.0001
cfg.train.gradient_thresh = 2000.0
cfg.train.seed = 555

cfg.train_category_feat_dim = 1
cfg.train_bool_feat_dim = 1
cfg.train.feature_dim = 128
cfg.train.output_dim = 64
cfg.train.num_decision_steps = 6
cfg.train.relaxation_factor = 1.5
cfg.train.batch_momentum = 0.7
cfg.train.virtual_batch_size = 256
cfg.train.num_classes = 1
cfg.train.encoder_type = 'classification'

cfg.train.quarter_size = 100

cfg.train.app_data_batch_size = 40000

cfg.validation = EasyDict()
cfg.validation.batch_size = 16384

