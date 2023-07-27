from functools import partial
import jax
import jax.numpy as jnp
import numpy as np
from torch import Generator
from torch.utils.data import DataLoader


def deep_update(dict1, dict2):
  updated_mapping = dict1.copy()
  for k, v in dict2.items():
    if k in updated_mapping and isinstance(updated_mapping[k], dict) \
       and isinstance(v, dict):
      updated_mapping[k] = deep_update(updated_mapping[k], v)
    else:
      updated_mapping[k] = v
  return updated_mapping


def sample_gaussian(rng, mean, logvar):
  normal = jax.random.normal(rng, shape=mean.shape, dtype=mean.dtype)
  return mean + jnp.exp(logvar / 2) * normal


def gumbel_softmax(rng: jnp.ndarray,
                   logits: jnp.ndarray,
                   tau: float,
                   scale: float,
                   hard: bool,
                   axis: int = -1) -> jnp.ndarray:
  def straight_through(y_soft):
      index = jnp.argmax(y_soft, axis=axis)
      y_hard = jax.nn.one_hot(index, logits.shape[-1])
      return y_hard - jax.lax.stop_gradient(y_soft) + y_soft
  gumbels = jax.random.gumbel(rng, logits.shape, logits.dtype)
  logits_sample = logits + gumbels * scale
  y_soft = jax.nn.softmax(logits_sample / tau, axis=axis)
  out = jax.lax.cond(hard, straight_through, lambda x: x, y_soft)
  return out, logits_sample


def cosine_anneal(step, start_value, final_value, start_step, final_step):
  if step < start_step:
    value = start_value
  elif step >= final_step:
    value = final_value
  else:
    a = 0.5 * (start_value - final_value)
    b = 0.5 * (start_value + final_value)
    progress = (step - start_step) / (final_step - start_step)
    value = a * np.cos(np.pi * progress) + b
  return value


class NumpyLoader(DataLoader):
  def __init__(self, dataset, seed=None,
               batch_size_device=1, num_devices=1,
               shuffle=False, sampler=None,
               batch_sampler=None, num_workers=0,
               pin_memory=False, drop_last=False,
               timeout=0, worker_init_fn=None):
    if seed is not None:
      self.seed = seed
      generator = Generator()
      generator.manual_seed(seed)
    else:
      generator = None
    super(self.__class__, self).__init__(
        dataset,
        batch_size=batch_size_device * num_devices,
        shuffle=shuffle,
        sampler=sampler,
        batch_sampler=batch_sampler,
        num_workers=num_workers,
        collate_fn=partial(
            numpy_collate,
            batch_size=batch_size_device,
            num_devices=num_devices),
        pin_memory=pin_memory,
        drop_last=drop_last,
        timeout=timeout,
        worker_init_fn=worker_init_fn,
        generator=generator)
    self.batch_size_device = batch_size_device
    self.num_devices = num_devices


def numpy_collate(batch, batch_size, num_devices):
  if isinstance(batch[0], np.ndarray):
    out = np.stack(batch)
    return out.reshape((num_devices, batch_size) + out.shape[1:])
  elif isinstance(batch[0], (tuple, list)):
    transposed = zip(*batch)
    return [numpy_collate(samples, batch_size, num_devices)
            for samples in transposed]
  else:
    out = np.array(batch)
    return out.reshape((num_devices, batch_size) + out.shape[1:])
