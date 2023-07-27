import argparse
from box import Box
from copy import deepcopy
from einops import rearrange
import haiku as hk
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import pickle
from typing import NamedTuple

from data.ground_truth import named_data, util
from evaluation.metrics import mig, factor_vae, dci, beta_vae
from evaluation.metrics import modularity_explicitness, sap_score
# from evaluation.metrics import downstream_task
from utils import NumpyLoader, gumbel_softmax, sample_gaussian


#


parser = argparse.ArgumentParser()
parser.add_argument('--save_file', type=str, help='')

args = parser.parse_args()

name = args.save_file.split('/')[-1].split('.pkl')[0]


#


logging.basicConfig(format='%(asctime)s - %(levelname)s - '
                           '%(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO,
                    handlers=[
                        # logging.FileHandler(f'{name}.log'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)
logger.info(f'Starting evaluation {name}')


#


class TrainState(NamedTuple):
  params: hk.Params
  params_disc: hk.Params or None
  state: hk.State
  opt_state: optax.OptState
  opt_state_disc: optax.OptState or None


#


load_file = args.save_file
with open(load_file, 'rb') as f:
  status = pickle.load(f)
config_dict = status['config']

config = Box(config_dict, frozen_box=True)
logger.info(f'{config}')


#


dataset = named_data.get_named_ground_truth_data(config.dataset.dataset_name)


#


def make_init_embed_fn(config):
  unif = jnp.linspace(-1.0, 1.0, config.model.K)[:, None]
  return hk.initializers.Constant(unif)


#


def encode(x, config):
  for i in range(2):
      x = hk.Conv2D(32, (4, 4), (2, 2), name=f'enc_conv{i}')(x)
      x = jax.nn.relu(x)
  for i in range(2, 4):
      x = hk.Conv2D(64, (2, 2), (2, 2), name=f'enc_conv{i}')(x)
      x = jax.nn.relu(x)
  x = rearrange(x, '... h w d -> ... (h w d)')
  x = hk.Linear(config.model.hidden, name='enc_mlp')(x)
  x = jax.nn.relu(x)
  x = hk.Linear(config.model.N * config.model.K, name='enc_attn_key')(x)
  x = rearrange(x, '... (n k) -> ... n k',
                n=config.model.N, k=config.model.K)
  return x


def encode_gaussian(x, config):
  for i in range(2):
      x = hk.Conv2D(32, (4, 4), (2, 2), name=f'enc_conv{i}')(x)
      x = jax.nn.relu(x)
  for i in range(2, 4):
      x = hk.Conv2D(64, (2, 2), (2, 2), name=f'enc_conv{i}')(x)
      x = jax.nn.relu(x)
  x = rearrange(x, '... h w d -> ... (h w d)')
  x = hk.Linear(config.model.hidden, name='enc_mlp')(x)
  x = jax.nn.relu(x)
  means = hk.Linear(config.model.N, name='enc_means')(x)
  log_var = hk.Linear(config.model.N, name='enc_logvar')(x)
  return means, log_var


def decode(z, config):
  z = hk.nets.MLP(
      [config.model.hidden, 1024],
      activation=jax.nn.relu,
      activate_final=True,
      name='dec_mlp')(z)
  z = rearrange(z, 'b (h w d) -> b h w d', h=4, w=4, d=64)
  z = hk.Conv2DTranspose(64, (4, 4), (2, 2), name='dec_deconv0')(z)
  z = jax.nn.relu(z)
  for i in range(1, 3):
      z = hk.Conv2DTranspose(32, (4, 4), (2, 2), name=f'dec_deconv{i}')(z)
      z = jax.nn.relu(z)
  z = hk.Conv2DTranspose(config.dataset.num_channels, (4, 4), (2, 2),
                         name='dec_deconv3')(z)
  return z


def discretize(logits, embed, config):
  scale = hk.get_state(
      'scale', shape=[], dtype=logits.dtype,
      init=hk.initializers.Constant(config.discrete.scale.init))
  lam = hk.get_state(
      'lambda', shape=[], dtype=logits.dtype,
      init=hk.initializers.Constant(1.0))
  hard = hk.get_state('hard', shape=[], dtype=logits.dtype,
                      init=hk.initializers.Constant(False))
  attn, sample = gumbel_softmax(
      hk.next_rng_key(), logits, lam, scale=scale, hard=hard)
  x = jnp.einsum('...nk,nkd->...nd', attn, embed)
  return rearrange(x, 'b n d -> b (n d)'), sample


def _forward(input, config):
  logits = encode(input, config)
  embed = hk.get_parameter(
      'embed',
      shape=[config.model.N, config.model.K, 1],
      dtype=input.dtype,
      init=make_init_embed_fn(config))
  embed = jax.lax.stop_gradient(embed)
  z, sample = discretize(logits, embed, config)
  recon = decode(z, config)
  return {
      'recon': recon,
      'logits': logits,
      'z': z
  }


def _forward_gaussian(input, config):
  logits = encode_gaussian(input, config)
  means, log_var = logits
  z = sample_gaussian(hk.next_rng_key(), means, log_var)
  recon = decode(z, config)
  return {
      'recon': recon,
      'logits': logits,
      'z': z
  }


def _representation_fn(input, config):
  attn_logits = encode(input, config)
  embed = hk.get_parameter(
      'embed',
      shape=[config.model.N, config.model.K, 1],
      dtype=input.dtype,
      init=make_init_embed_fn(config))
  attn_soft = jax.nn.softmax(attn_logits, axis=-1)
  z = jnp.einsum('...nk,nkd->...nd', attn_soft, embed)
  z = rearrange(z, '... n d -> ... (n d)')
  return z


def _representation_fn_gaussian(input, config):
  means, log_var = encode_gaussian(input, config)
  return means


def _discriminator(sample, config):
  x = hk.nets.MLP(
      output_sizes=[1000] * config.discriminator.num_layers + [2],
      activation=jax.nn.leaky_relu,
      activate_final=False,
      name='discriminator')(sample)
  return x


#


if config.run.discrete:
  forward = hk.transform_with_state(_forward)
  rep_fn = hk.without_apply_rng(hk.transform(_representation_fn))
else:
  forward = hk.transform_with_state(_forward_gaussian)
  rep_fn = hk.without_apply_rng(hk.transform(_representation_fn_gaussian))
if config.loss.tc:
  discriminator = hk.without_apply_rng(hk.transform(_discriminator))


#


def sigmoid_binary_cross_entropy(logits, labels):
  return jnp.clip(logits, a_min=0) - logits * labels + \
      jnp.log(1 + jnp.exp(-jnp.abs(logits)))


def bce_loss(logits, labels):
  return jnp.sum(sigmoid_binary_cross_entropy(logits, labels), [1, 2, 3])


def gumbel_kl_loss(logits):
  K = logits.shape[-1]
  log_q_y = jax.nn.log_softmax(logits, -1)
  q_y = jnp.exp(log_q_y)
  kl_tmp = q_y * (log_q_y - jnp.log(1.0 / K))
  return jnp.sum(kl_tmp, -1)


def gaussian_kl_loss(logits):
  mean, logvar = logits
  return 0.5 * (jnp.square(mean) + jnp.exp(logvar) - logvar - 1)


def disc_tc_loss(params, sample, config):
  logits = discriminator.apply(params, sample, config)
  return logits[:, 0] - logits[:, 1]


def loss_fn(params, params_disc, state, rng, input, config):
  out, state = forward.apply(params, state, rng, input, config)
  recon_loss = bce_loss(out['recon'], input)
  kl_loss_fn = gumbel_kl_loss if config.run.discrete else gaussian_kl_loss
  kl_loss_tmp = kl_loss_fn(out['logits'])
  kl_loss = jnp.sum(kl_loss_tmp, -1)
  loss = recon_loss
  loss += kl_loss
  if config.loss.tc:
    tc_loss = disc_tc_loss(params_disc, out['z'], config)
    loss += config.discriminator.gamma * tc_loss
  loss = jnp.mean(loss)
  losses = {
      'loss': loss,
      'recon_loss': jnp.mean(recon_loss),
      'kl_loss': jnp.mean(kl_loss),
      'elbo': jnp.mean(recon_loss + kl_loss),
  }
  if config.loss.tc:
    losses['tc_loss'] = jnp.mean(tc_loss)
  return loss, (losses, state, out['z'])


def shuffle_codes(rng, sample):
  z_shuffle = list()
  for i in range(sample.shape[-1]):
      rng, key = jax.random.split(rng)
      z_shuffle.append(jax.random.permutation(key, sample[:, i]))
  return jnp.stack(z_shuffle, 1)


def disc_loss_fn(params, rng, sample, config):
  sample_perm = shuffle_codes(rng, sample)
  logits = discriminator.apply(params, sample, config)
  logits_fake = discriminator.apply(params, sample_perm, config)
  zeros = jnp.zeros(logits.shape[0], dtype=int)
  ones = jnp.ones(logits.shape[0], dtype=int)
  loss = optax.softmax_cross_entropy_with_integer_labels(logits, zeros)
  loss_f = optax.softmax_cross_entropy_with_integer_labels(logits_fake, ones)
  return jnp.mean(0.5 * (loss + loss_f))


# @partial(jax.jit, static_argnames=['config'])
def get_loss(train_state, rng, input, config):
  _, key = jax.random.split(rng)
  params, params_disc, state, opt_state, opt_state_disc = train_state
  _, (scalars, _, sample) = loss_fn(
      params, params_disc, state, rng, input, config)
  if config.loss.tc:
    loss_disc = disc_loss_fn(params_disc, key, sample, config)
    scalars['disc_loss'] = loss_disc
  return scalars


# @partial(jax.jit, static_argnames=['config'])
def represent(train_state, input, config):
  params, _, _, _, _ = train_state
  return rep_fn.apply(params, input, config)


def evaluation(dataset_eval, train_state, config):
  def rep_function(input):
    return represent(train_state, input, config)

  random_seed = np.random.RandomState(42)
  eval_dict = dict()
  eval_dict.update(mig.compute_mig(
      dataset_eval, rep_function,
      random_state=random_seed, num_train=10000, num_bins=20))
  eval_dict.update(modularity_explicitness.compute_modularity_explicitness(
      dataset_eval, rep_function, random_seed))
  eval_dict.update(sap_score.compute_sap(
      dataset_eval, rep_function, random_seed))
  eval_dict.update(factor_vae.compute_factor_vae(
      dataset_eval, rep_function, random_seed))
  eval_dict.update(beta_vae.compute_beta_vae_sklearn(
      dataset_eval, rep_function, random_seed))
  eval_dict.update(dci.compute_dci(
      dataset_eval, rep_function, random_seed))

  # eval_dict.update(downstream_task.compute_downstream_task(
  #     dataset_eval, rep_function, 'logistic_regression', random_seed))
  # eval_dict.update(downstream_task.compute_downstream_task(
  #     dataset_eval, rep_function, 'gradient_boosting', random_seed))
  return eval_dict


#


def update_hard_beta(train_state, hard, beta):
  params, params_disc, state, opt_state, opt_state_disc = train_state
  state2 = deepcopy(state)
  state2['~']['hard'] = hard
  state2['~']['scale'] = beta
  return TrainState(params, params_disc, state2, opt_state, opt_state_disc)


#


def evaluate(status):
  train_state = status['train_state']
  rng = status['rng']
  random_state = np.random.RandomState(42)

  dataset_train = util.tf_data_set_from_ground_truth_data(
      dataset,
      random_state.randint(2**32))
  data_loader = NumpyLoader(dataset_train,
                            batch_size_device=config.run.batch_size,
                            num_devices=1)
  data = iter(data_loader)

  if config.run.discrete:
    train_state_hard = update_hard_beta(train_state, True, 0.)
    train_state = update_hard_beta(train_state, False, 0.)

  eval_dict = evaluation(dataset, train_state, config)

  gamma = config.discriminator.gamma if config.loss.tc else 0.
  vae = 'dvae' if config.run.discrete else 'vae'
  model = vae if not config.loss.tc else 'factor_' + vae

  res = {
      'dataset': f"'{config.dataset.dataset_name}'",
      'model': model,
      'param_name': 'gamma',
      'param': gamma,
      'mig': eval_dict['mig_20bins'],
      'factor_vae_metric': eval_dict['eval_accuracy'],
      'dci_disentanglement': eval_dict['disentanglement'],
      'beta_vae_sklearn': eval_dict['beta_eval_accuracy'],
      'modularity': eval_dict['modularity_score'],
      'sap_score': eval_dict['SAP_score'],

      # 'downstream_task_bt10': eval_dict['bt10:mean_test_accuracy'],
      # 'downstream_task_bt100': eval_dict['bt100:mean_test_accuracy'],
      # 'downstream_task_bt1000':
      #     eval_dict['bt1000:mean_test_accuracy'],
      # 'downstream_task_bt10000':
      #     eval_dict['bt10000:mean_test_accuracy'],
      # 'downstream_task_lg10': eval_dict['bt10:mean_test_accuracy'],
      # 'downstream_task_lg100': eval_dict['bt100:mean_test_accuracy'],
      # 'downstream_task_lg1000':
      #     eval_dict['lg1000:mean_test_accuracy'],
      # 'downstream_task_lg10000':
      #     eval_dict['lg10000:mean_test_accuracy']
  }

  if config.run.discrete:
    losses_dict = dict()
    for _ in range(1000):
      rng, key = jax.random.split(rng)
      batch = next(data)[0]
      losses = get_loss(train_state, key, batch, config)
      for loss, value in losses.items():
        losses_dict[loss] = losses_dict.get(loss, []) + [jax.device_get(value)]
      losses_hard = get_loss(train_state_hard, key, batch, config)
      for loss, value in losses_hard.items():
        loss_h = loss + '_st'
        loss_h_val = losses_dict.get(loss_h, []) + [jax.device_get(value)]
        losses_dict[loss_h] = loss_h_val
    loss_names = list(losses_dict.keys())
    for loss in loss_names:
      losses_dict[loss] = np.mean(losses_dict[loss], 0)

    st_gap = abs(losses_dict['loss_st'] - losses_dict['loss'])
    res['st_gap'] = st_gap

  logger.info(res)


#


evaluate(status)
