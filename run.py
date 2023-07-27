import argparse
from box import Box
from einops import rearrange
from functools import partial
import haiku as hk
import jax
import jax.numpy as jnp
import logging
import numpy as np
import optax
import pickle
from typing import NamedTuple

from data.ground_truth import named_data, util
from evaluation.metrics import mig
from utils import NumpyLoader, deep_update, cosine_anneal
from utils import gumbel_softmax, sample_gaussian


#


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, help='')
parser.add_argument('--gaussian', type=bool, nargs='?', const=True,
                    default=False, help='')
parser.add_argument('--K', type=int, nargs='?', const=64, default=64,
                    help='')
parser.add_argument('--init', type=float, nargs='?', const=.5,
                    default=.5, help='')
parser.add_argument('--final', type=float, nargs='?', const=2.,
                    default=2., help='')
parser.add_argument('--tc', type=bool, nargs='?', const=True,
                    default=False, help='')
parser.add_argument('--gamma', type=float, nargs='?', const=.0,
                    default=.0, help='')
parser.add_argument('--seed', type=int, nargs='?', const=42, default=42,
                    help='')
parser.add_argument('--save_path', type=str, default='./', help='')


args = parser.parse_args()


#


dataset = named_data.get_named_ground_truth_data(args.dataset)


#


config_shared = {
    'run': {
        'batch_size': 64,
        'num_train_steps': 300000,
        'log_every': 1000,
        'eval_every': 10000,
        'save_every': 50000,
    },
    'model': {
        'N': 10,
        'hidden': 256
    },
    'loss': {
        'lr': 1e-4,
    },
    'discriminator': {
        'num_layers': 6,
        'optimizer': {
            'lr': 1e-4,
            'b1': .5,
            'b2': .9
        }
    }
}


#


config = {
    'run': {
        'discrete': not args.gaussian,
        'seed': args.seed,
        'save_path': args.save_path
    },
    'loss': {
        'tc': args.tc
    },
    'dataset': {
        'dataset_name': args.dataset,
        'decode_res': tuple(dataset.observation_shape[:2]),
        'num_channels': dataset.observation_shape[-1],
        'factor_sizes': dataset.factors_num_values
    },
    'model': {
        'K': args.K
    },
    'discrete': {
        'scale': {
            'init': args.init,
            'final': args.final
        }
    },
    'discriminator': {
        'gamma': args.gamma
    }
}


#


def update_config(config):
  config = deep_update(config_shared, config)
  if config['run']['discrete'] is False:
    del config['discrete']
    del config['model']['K']
  if config['loss']['tc'] is False:
    del config['discriminator']
  return config


#


config = update_config(config)
vae = '_gauss' if args.gaussian else ''
factor = f'_factor{args.gamma}' if args.tc else ''
name = f'{args.dataset}{vae}{factor}'


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
logger.info(f'Starting run {name}')
logger.info(config)

config = Box(config, frozen_box=True)


#


class TrainState(NamedTuple):
    params: hk.Params
    params_disc: hk.Params or None
    state: hk.State
    opt_state: optax.OptState
    opt_state_disc: optax.OptState or None


#


def make_init_embed_fn(config):
  unif = jnp.linspace(-1.0, 1.0, config.model.K)[:, None]
  return hk.initializers.Constant(unif)


def make_optimizer(config):
  return optax.adam(learning_rate=config.loss.lr)


def make_optimizer_disc(config):
  return optax.adam(
      learning_rate=config.discriminator.optimizer.lr,
      b1=config.discriminator.optimizer.b1,
      b2=config.discriminator.optimizer.b2)


#


def anneal_temperature(step, config):
  anneal_fn = cosine_anneal
  scale_new = anneal_fn(
      step,
      config.discrete.scale.init,
      config.discrete.scale.final,
      0.,
      config.run.num_train_steps
  )
  return scale_new


def update_temperatures(train_state, scale):
  params, params_disc, state, opt_state, opt_state_disc = train_state
  state['~']['scale'] = scale
  return TrainState(params, params_disc, state, opt_state, opt_state_disc)


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


optimizer = make_optimizer(config)
if config.run.discrete:
  forward = hk.transform_with_state(_forward)
  rep_fn = hk.without_apply_rng(hk.transform(_representation_fn))
else:
  forward = hk.transform_with_state(_forward_gaussian)
  rep_fn = hk.without_apply_rng(hk.transform(_representation_fn_gaussian))
if config.loss.tc:
  discriminator = hk.without_apply_rng(hk.transform(_discriminator))
  optimizer_disc = make_optimizer_disc(config)


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


@partial(jax.jit, static_argnames=['config'])
def train_step(train_state, rng, input, config):
  _, key = jax.random.split(rng)
  params, params_disc, state, opt_state, opt_state_disc = train_state
  grads, (scalars, new_state, sample) = jax.grad(loss_fn, has_aux=True)(
      params, params_disc, state, rng, input, config)
  updates, new_opt_state = optimizer.update(grads, opt_state)
  new_params = optax.apply_updates(params, updates)
  if config.loss.tc:
    loss_disc, grads_disc = jax.value_and_grad(disc_loss_fn)(
        params_disc, key, sample, config)
    updates_disc, new_opt_state_disc = optimizer_disc.update(
        grads_disc, opt_state_disc)
    new_params_disc = optax.apply_updates(params_disc, updates_disc)
    scalars['disc_loss'] = loss_disc
    train_state_new = TrainState(new_params, new_params_disc, new_state,
                                 new_opt_state, new_opt_state_disc)
  else:
    train_state_new = TrainState(new_params, None, new_state,
                                 new_opt_state, None)
  return train_state_new, scalars


def initial_state(rng, input, config):
  _, key = jax.random.split(rng)
  params, state = forward.init(rng, input, config)
  opt_state = optimizer.init(params)
  if config.loss.tc:
    out, _ = forward.apply(params, state, rng, input, config)
    params_disc = discriminator.init(key, out['z'], config)
    opt_state_disc = optimizer_disc.init(params_disc)
    return TrainState(params, params_disc, state, opt_state, opt_state_disc)
  else:
    return TrainState(params, None, state, opt_state, None)


#


@partial(jax.jit, static_argnames=['config'])
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
  return eval_dict


#


def run(config):

  seed = config.run.seed
  rng = jax.random.PRNGKey(seed)
  random_state = np.random.RandomState(42)

  dataset_train = util.tf_data_set_from_ground_truth_data(
      dataset,
      random_state.randint(2**32))
  data_loader = NumpyLoader(dataset_train,
                            batch_size_device=config.run.batch_size)
  data = iter(data_loader)

  batch = next(data)[0]

  rng, key = jax.random.split(rng)
  train_state = initial_state(key, batch, config)

  for step_num in range(1, config.run.num_train_steps + 1):

    rng, key = jax.random.split(rng)
    batch = next(data)[0]
    train_state, train_scalars = train_step(
        train_state, key, batch, config)

    extras = {}
    if config.run.discrete:
      scale = anneal_temperature(step_num, config)
      train_state = update_temperatures(train_state, scale)
      extras = {'scale': np.round(scale, 4)}

    if step_num % config.run.log_every == 0:
      train_means = jax.tree_map(
          lambda v: np.round(v.item(), 4), jax.device_get(train_scalars))
      metrics = {**train_means, **extras}
      logger.info(metrics)

    if step_num % config.run.eval_every == 0:
      eval_dict = evaluation(dataset, train_state, config)
      mig_dict = {
          'mig': eval_dict['mig_20bins'],
      }
      logger.info(mig_dict)

    if step_num % config.run.save_every == 0:
      status = {'step_num': step_num,
                'train_state': train_state,
                'rng': rng,
                'config': config}
      root = config.run.save_path
      save_file = root + '{}.pkl'.format(name)
      with open(save_file, 'wb') as f:
        pickle.dump(status, f)

  #

  status = {'step_num': step_num,
            'train_state': train_state,
            'rng': rng,
            'config': config}
  root = config.run.save_path
  save_file = root + '{}.pkl'.format(name)
  with open(save_file, 'wb') as f:
    pickle.dump(status, f)


#


run(config)
