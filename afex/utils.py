r"""AFEX utilities."""
# Authors: Zilin Song.


import io
import os
import abc
import typing

import ml_collections
import jax.numpy as jnp
import     numpy as  np

import alphafold.common.protein as _af_protein
import alphafold.model.utils    as _af_utils


# Type hints.
TAFFeatures = dict[str, typing.Any]
TAFConfig   = ml_collections.ConfigDict
TAFParams   = typing.Mapping[str, typing.Mapping[str, jnp.ndarray]]
TAFResults  = dict[str, dict[str, jnp.ndarray]]
TAFProtein  = _af_protein.Protein


# Directories.
AF_PARAMS_DIR = '/u/songzl/3.alphafoldtools/params'


# Parameters loader.
def load_params(model_name: str):
  with open(os.path.join(AF_PARAMS_DIR, f"params_{model_name}.npz"), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  params = _af_utils.flat_params_to_haiku(params=params)
  return params