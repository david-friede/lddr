# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Downstream classification task."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from evaluation.metrics import utils
import logging
import numpy as np
from six.moves import range

logger = logging.getLogger(__name__)


def compute_downstream_task(ground_truth_data,
                            representation_function,
                            predictor_fn,
                            random_state,
                            num_train=[10, 100, 1000, 10000],
                            num_test=5000,
                            artifact_dir=None,
                            batch_size=16):
  """Computes loss of downstream task.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    num_train: Number of points used for training.
    num_test: Number of points used for testing.
    batch_size: Batch size for sampling.

  Returns:
    Dictionary with scores.
  """
  del artifact_dir
  scores = {}
  for train_size in num_train:
    logger.info(f'train_size {train_size}')
    mus_train, ys_train = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, train_size, random_state,
        batch_size)
    mus_test, ys_test = utils.generate_batch_factor_code(
        ground_truth_data, representation_function, num_test, random_state,
        batch_size)

    train_err, test_err = _compute_loss(
        np.transpose(mus_train), ys_train, np.transpose(mus_test),
        ys_test, predictor_fn)
    size_string = str(train_size)
    pred_string = 'lg' if predictor_fn == 'logistic_regression' else 'bt'
    scores[pred_string + size_string +
           ":mean_train_accuracy"] = np.mean(train_err)
    scores[pred_string + size_string +
           ":mean_test_accuracy"] = np.mean(test_err)
    # scores[size_string +
    #        ":min_train_accuracy"] = np.min(train_err)
    # scores[size_string + ":min_test_accuracy"] = np.min(test_err)
    # for i in range(len(train_err)):
    #   scores[size_string +
    #          ":train_accuracy_factor_{}".format(i)] = train_err[i]
    #   scores[size_string +
    #          ":test_accuracy_factor_{}".format(i)] = test_err[i]

  return scores


def clean_input(x_train, y_train):
  size = x_train.shape[0]
  _, u_index, u_count = np.unique(
      y_train, return_index=True, return_counts=True)
  filter_index = u_index[u_count == 1]
  index = np.setdiff1d(np.arange(size), filter_index)
  max_splits = u_count[u_count != 1].min() if len(u_count[u_count != 1]) else 0
  return x_train[index], y_train[index], max_splits


def _compute_loss(x_train, y_train, x_test, y_test, predictor_fn):
  """Compute average accuracy for train and test set."""
  num_factors = y_train.shape[0]
  train_loss = []
  test_loss = []
  for i in range(num_factors):
    x, y, max_splits = clean_input(x_train, y_train[i, :])
    if max_splits == 0:
      continue
    elif len(np.unique(y)) == 1:
      continue
    else:
      n_splits = min(5, max_splits)

      predictor_model = utils.make_predictor_fn(predictor_fn, n_splits)
      model = predictor_model()

      model.fit(x, y)
      train_loss.append(np.mean(model.predict(x) == y))
      test_loss.append(np.mean(model.predict(x_test) == y_test[i, :]))
  return train_loss, test_loss
