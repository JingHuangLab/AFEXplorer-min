r"""AFEX utilities."""
# Authors: Zilin Song.


import io
import os
import abc
import typing

import ml_collections
import jax.numpy as jnp
import     numpy as  np

import alphafold.common.protein as _af_prot
import alphafold.model.utils    as _af_utils


# Type hints.
TAFFeatures = dict[str, typing.Any]
TAFConfig   = ml_collections.ConfigDict
TAFParams   = typing.Mapping[str, typing.Mapping[str, jnp.ndarray]]
TAFResults  = dict[str, dict[str, jnp.ndarray]]
TAFProtein  = _af_prot.Protein


# Directories.
AF_PARAMS_DIR = '/u/songzl/3.alphafoldtools/params'


# Parameters loader.
def load_params(model_name: str):
  with open(os.path.join(AF_PARAMS_DIR, f"params_{model_name}.npz"), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  params = _af_utils.flat_params_to_haiku(params=params)
  return params


def cast_jnp_to_np(jnp_array_dict: TAFResults) -> dict[str, dict[str, np.ndarray]]:
  """Recursively cast in the results dict all jax arrays to numpy arrays."""
  for k, v in jnp_array_dict.items():
    if   isinstance(v, dict):
      jnp_array_dict[k] = cast_jnp_to_np(v)
    elif isinstance(v, jnp.ndarray):
      jnp_array_dict[k] = np.asarray(v)
  return jnp_array_dict 