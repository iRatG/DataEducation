# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TabNet model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf




def glu(act, n_units):
  """=====================Generalized linear unit nonlinear activation."""
# Сигмовидная функция страдает от проблемы «исчезающих градиентов», поскольку 
# она сглаживается на обоих концах, что приводит к очень небольшим изменениям веса 
# при обратном распространении. Это может заставить нейронную сеть отказаться 
# учиться и застрять. По этой причине использование сигмоидальной функции заменяется 
# другими нелинейными функциями, такими как выпрямленная линейная единица (ReLU).
  return act[:, :n_units] * tf.nn.sigmoid(act[:, n_units:])


class TabNet(object):
  """=====================TabNet model class."""

  def __init__(self,
               columns,
               num_features,
               feature_dim,
               output_dim,
               num_decision_steps,
               relaxation_factor,
               batch_momentum,
               virtual_batch_size,
               num_classes,
               epsilon=0.00001):
    """Initializes a TabNet instance.
    Args:
      columns: The Tensorflow column names for the dataset.

      num_features: The number of input features (i.e the number of columns for
        tabular data assuming each feature is represented with 1 dimension).

      feature_dim: Dimensionality of the hidden representation in feature
        transformation block. Each layer first maps the representation to a
        2*feature_dim-dimensional output and half of it is used to determine the
        nonlinearity of the GLU activation where the other half is used as an
        input to GLU, and eventually feature_dim-dimensional output is
        transferred to the next layer.

      output_dim: Dimensionality of the outputs of each decision step, which is
        later mapped to the final classification or regression output.

      num_decision_steps: Number of sequential decision steps.

      relaxation_factor: Relaxation factor that promotes the reuse of each
        feature at different decision steps. When it is 1, a feature is enforced
        to be used only at one decision step and as it increases, more
        flexibility is provided to use a feature at multiple decision steps.

      batch_momentum: Momentum in ghost batch normalization.

      virtual_batch_size: Virtual batch size in ghost batch normalization. The
        overall batch size should be an integer multiple of virtual_batch_size.

      num_classes: Number of output classes.

      epsilon: A small number for numerical stability of the entropy calcations.
    Returns:
      A TabNet instance.
    """

    self.columns = columns
    self.num_features = num_features
    self.feature_dim = feature_dim
    self.output_dim = output_dim
    self.num_decision_steps = num_decision_steps
    self.relaxation_factor = relaxation_factor
    self.batch_momentum = batch_momentum
    self.virtual_batch_size = virtual_batch_size
    self.num_classes = num_classes
    self.epsilon = epsilon

  def encoder(self, data, reuse, is_training):
    """===========TabNet encoder model."""
    # Контекстный менеджер для определения операций, создающих переменные (активация слоя).
    with tf.variable_scope("Encoder", reuse=reuse):

      #===========Reads and normalizes input features.
      # Чтение входящих данных.
      features = tf.feature_column.input_layer(data, self.columns)
      # Нормализация входящих данных. Делаем это хорошо. Без виртуальностея. Моментум и Виртуальный батч не используем.
      features = tf.layers.batch_normalization(
          features, training=is_training, momentum=self.batch_momentum)
      # Возвращает тензор, содержащий размерность входного тензора.
      batch_size = tf.shape(features)[0]

      # ===============Initializes decision-step dependent variables.\ Инициализация данных.
      # Перенимаем выходную структуру нужной размерности слоя и заполняем нулями
      output_aggregated = tf.zeros([batch_size, self.output_dim])
      # передаем в маску нормализованные данные.
      masked_features = features
      # заполняем нулями вектор с заданной размерностью.
      mask_values = tf.zeros([batch_size, self.num_features])
      # заполняем нулями вектор, который будет отвечать за агрегацию 
      aggregated_mask_values = tf.zeros([batch_size, self.num_features])
      # заполняем единицами такой же вектор, заданной размерности
      complemantary_aggregated_mask_values = tf.ones(
          [batch_size, self.num_features])
      # показатель энропии = 0
      total_entropy = 0

      # Если обучаемся, задаем параметр входного виртаульного батча. Если нет, то будет один проход.
      if is_training:
        v_b = self.virtual_batch_size
      else:
        v_b = 1

      for ni in range(self.num_decision_steps):

        # Feature transformer with two shared and two decision step dependent
        # blocks is used below.

        reuse_flag = (ni > 0)
        # Первый проход будет инициализирующим.
        # Первый блок. FC -> BN -> GLU

        transform_f1 = tf.layers.dense( # Создаем слой.
            masked_features,            # Принимаем на вход нормализованные данные.
            self.feature_dim * 2,       # Размерность скрытого представления в блоке преобразования объектов. 
                                        # Каждый слой сначала отображает представление на 2*feature_dim-мерный выход, 
                                        # и половина его используется для определения нелинейности активации GLU, 
                                        # где другая половина используется в качестве входного сигнала для GLU, 
                                        # и в конечном итоге feature_dim-мерный выход передается следующему слою.
            name="Transform_f1",        # Имя слоя
            reuse=reuse_flag,           # Переиспользовать переменные
            use_bias=False)             # Без использования вектора

        transform_f1 = tf.layers.batch_normalization( # Нормализуем данные.  Для повышения производительности при увеличении серий все 
                                                      # операции BN, за исключением той, которая применяется к входным функциям, 
                                                      # реализуются в форме ложного BN с виртуальным размером серии BV и импульсом mB
            transform_f1,                             # Весь слой передаем с Full Conected Layer..
            training=is_training,                     # Сообщаем, что обучаемся, чтобы запомнить все.
            momentum=self.batch_momentum,             # Импульс. Показатель для нормализации.
            virtual_batch_size=v_b)                   # размер батча

        transform_f1 = glu(transform_f1, self.feature_dim) # применяем GLU и задаем размерность изначальную. 
        # На этом первый блок закончен

        # Второй блок. FC -> BN -> GLU Сумма под корнем.
        transform_f2 = tf.layers.dense( # Создаем слой
            transform_f1,               # Принимаем данные с первого слоя  
            self.feature_dim * 2,       # Увеличиваем размерность.
            name="Transform_f2",        # Имя слоя 
            reuse=reuse_flag,           # Переиспользовать переменные
            use_bias=False)             # Без использования вектора

        transform_f2 = tf.layers.batch_normalization( # Нормализуем данные. Для повышения производительности при увеличении серий все 
                                                      # операции BN, за исключением той, которая применяется к входным функциям, 
                                                      # реализуются в форме ложного BN с виртуальным размером серии BV и импульсом mB

            transform_f2,                             # Принимаем данные с Full Conected Layer.
            training=is_training,                     # Cообщаем, что обучаемся, чтобы запоминать данные.
            momentum=self.batch_momentum,             # Импульс с которым будем нормализовать данные.
            virtual_batch_size=v_b)                   # Размер батча

        # Размерность уменьшаем до начальной. соединяем 2 набора и 1 и 2 после GLU. берем корень 
        transform_f2 = (glu(transform_f2, self.feature_dim) + transform_f1) * np.sqrt(0.5)


        # Теперь второй блок. Decision step  Зависимый
        transform_f3 = tf.layers.dense(               # Создаем слой
            transform_f2,                             # Принимаем на вход данные с первого блока
            self.feature_dim * 2,                     # Увеличиваем объем слоя
            name="Transform_f3" + str(ni),            # Наименование шага. Итерируемый номер.
            use_bias=False)                           # Без использования вектора.

        transform_f3 = tf.layers.batch_normalization( # Нормализуем выход Для повышения производительности при увеличении серий все 
                                                      # операции BN, за исключением той, которая применяется к входным функциям, 
                                                      # реализуются в форме ложного BN с виртуальным размером серии BV и импульсом mB
            transform_f3,                             # Принимаем данные с предыдущего соя.
            training=is_training,                     # Сообщаем, что обучаемся, чтобы запоминать данные.
            momentum=self.batch_momentum,             # Импульс с которым будем нормализовывать данные.
            virtual_batch_size=v_b)                   # Размер батча

        # Размерность уменьшаем до начальной. соединяем 2 набора и 3 и 2 после GLU. берем корень 
        transform_f3 = (glu(transform_f3, self.feature_dim) + transform_f2) * np.sqrt(0.5)

        # Второй шаг во втором блоке. Такой же как и предыдущий по логике.  
        transform_f4 = tf.layers.dense(
            transform_f3,
            self.feature_dim * 2,
            name="Transform_f4" + str(ni),
            use_bias=False)
        transform_f4 = tf.layers.batch_normalization(
            transform_f4,
            training=is_training,
            momentum=self.batch_momentum,
            virtual_batch_size=v_b)
        transform_f4 = (glu(transform_f4, self.feature_dim) + transform_f3) * np.sqrt(0.5)

        # Если Количество последовательных шагов принятия решения > 0
        # Первый проход будет инициализирующим, поэтому сюда не войдет.
        if ni > 0:
          # Первый расчет из выхода Feature Transform подаем на ReLU  
          decision_out = tf.nn.relu(transform_f4[:, :self.output_dim])

          # Decision aggregation.
          output_aggregated += decision_out # Соединяем все расчеты между собой и они пойдут на выход FC

          # Aggregated masks are used for visualization of the feature importance attributes.
          # Агрегированные маски используются для визуализации атрибутов важности объектов.
          scale_agg = tf.reduce_sum(decision_out, axis=1, keep_dims=True) / (self.num_decision_steps - 1)
          aggregated_mask_values += mask_values * scale_agg

        features_for_coef = (transform_f4[:, self.output_dim:])

        if ni < self.num_decision_steps - 1:
          # ===================ATTENTIVE TRANSFORMER=====================
          # FC -> BN - SPARSEMAX - SCALE
          # Determines the feature masks via linear and nonlinear
          # transformations, taking into account of aggregated feature use.

          mask_values = tf.layers.dense(        # Создание слоя Fully Connected
              features_for_coef,                # Данные из FeatTrans блока
              self.num_features,                # Входные фичи. количество
              name="Transform_coef" + str(ni),  # Коэффициент трансформации
              use_bias=False)                   # Без использования вектора.

          mask_values = tf.layers.batch_normalization( # Виртуальная нормализация.
              mask_values,                             # Данные с предыдущего слоя.
              training=is_training,                    # Сообщаем, что обучаемся
              momentum=self.batch_momentum,            # Данные для мнимой нормализации
              virtual_batch_size=v_b)

          # Перемножаем. С Prior Scales. Предыдущего хода.
          mask_values *= complemantary_aggregated_mask_values
          # Применяем SPARSEMAX, not SOFTMAX
          mask_values = tf.contrib.sparsemax.sparsemax(mask_values)

          # Relaxation factor controls the amount of reuse of features between
          # different decision blocks and updated with the values of coefficients.
          
          # Коэффициент релаксации управляет количеством повторного использования признаков 
          # между различными блоками принятия решений и обновляется значениями коэффициентов.
          complemantary_aggregated_mask_values *= (self.relaxation_factor - mask_values)

          # ================Блок ATTENTIVE TRANSFORMER завершен.================

          
          # Entropy is used to penalize the amount of sparsity in feature selection.
          # Энтропия используется, чтобы наказать количество разреженности в выборе признаков.

          total_entropy += tf.reduce_mean(
              tf.reduce_sum(
                  -mask_values * tf.log(mask_values + self.epsilon),
                  axis=1)) / (
                      self.num_decision_steps - 1)

          # Feature selection.
          masked_features = tf.multiply(mask_values, features)    # Отбор необходимых признаков

          # Visualization of the feature selection mask at decision step ni
          # Визуализация маски выбора объекта на этапе принятия решения ni
          tf.summary.image(
              "Mask for step" + str(ni),
              tf.expand_dims(tf.expand_dims(mask_values, 0), 3),
              max_outputs=1)

      # Visualization of the aggregated feature importances
      # Визуализация агрегированных значений признаков
      tf.summary.image(
          "Aggregated mask",
          tf.expand_dims(tf.expand_dims(aggregated_mask_values, 0), 3),
          max_outputs=1)

      return output_aggregated, total_entropy

  def classify(self, activations, reuse):
    """TabNet classify block."""
    # Контекстный менеджер для определения операций, создающих переменные (слои).
    with tf.variable_scope("Classify", reuse=reuse):
      # Создание слоя для проверки результата.
      logits = tf.layers.dense(activations, self.num_classes, use_bias=False)
      # Применение функции софтмакс для результатов слоя классификации.  
      predictions = tf.nn.softmax(logits)
      # Возврат результата последнего слоя и результат классификации
      return logits, predictions

  def regress(self, activations, reuse):
    """TabNet regress block."""

    with tf.variable_scope("Regress", reuse=reuse):
      predictions = tf.layers.dense(activations, 1)
      return predictions