r"""Shared utilities."""
# Authors: Zilin Song.


import io
import os
import copy
import typing
import functools

import ml_collections
import jax.numpy as jnp
import     numpy as  np

import alphafold.model.utils    as _af_utils
import alphafold.model.config   as _af_config
import alphafold.common.protein as _af_protein


# Type hints.
TAFFeatures = dict[str, typing.Any]
TAFConfig   = ml_collections.ConfigDict
TAFParams   = typing.Mapping[str, typing.Mapping[str, jnp.ndarray]]
TAFResults  = dict[str, dict[str, jnp.ndarray]]
TAFProtein  = _af_protein.Protein

AF_PARAMS_DIR = '/u/songzl/3.alphafoldtools/params'


# Monomer utils.
def _afex_monomer_config(model_name:    str,
                         msa_clusters:  int,
                         use_dropout:   bool, 
                         use_templates: bool, 
                         use_recycling: bool, 
                         use_remat:     bool, 
                         ) -> TAFConfig:
  # Load presets.
  assert model_name in _af_config.MODEL_PRESETS.get('monomer')
  config = copy.deepcopy(_af_config.CONFIG)
  config.update_from_flattened_dict(_af_config.CONFIG_DIFFS[model_name])
  # Data configs.  
  # NOTE: AF monomer preprocesses features in alphafold.model.RunModel().process_features(), so that
  #       config.data should be modified as well.
  # Turn off multimer_model.
  config.model.global_config.multimer_mode = False
  # Single ensemble.
  config.data.eval.num_ensemble = 1
  # Maybe reduce `num_msa_clusters` for efficient memory scaling.
  config.data.eval.max_msa_clusters = msa_clusters
  # If use dropout.
  config.model.global_config.deterministic = not use_dropout
  # If use template.
  config.data.eval.max_templates   = 4 if use_templates else 0
  config.data.common.use_templates = use_templates
  config.data.common.reduce_msa_clusters_by_max_templates = use_templates
  config.model.embeddings_and_evoformer.template.enabled  = use_templates
  # If use recycling.
  config.data.common.num_recycle = 3 if use_recycling else 0
  config.model.num_recycle       = 3 if use_recycling else 0
  config.model.resample_msa_in_recycling                 = use_recycling
  config.model.embeddings_and_evoformer.recycle_features = use_recycling
  config.model.embeddings_and_evoformer.recycle_pos      = use_recycling
  # If use re-materialization (reverse-mode autograd where necessary).
  config.model.global_config.use_remat = use_remat
  return config
def AFEXMonomerConfig(model_name:    str, 
                      msa_clusters:  int  = 512, 
                      use_dropout:   bool = False, 
                      use_templates: bool = False, 
                      use_recycling: bool = False, 
                      use_remat:     bool = True, 
                      ) -> typing.Callable[[], TAFConfig]:
  r"""The AFEX-monomer configuration.
  
    Args:
      model_name    (str): The name of the AF-monomer model.
      msa_clusters  (int): The number of MSA clusters, default: 512.
      use_dropout   (bool): If use dropout during forwarding, default: False.
      use_templates (bool): If embed templates, default: False.
      use_recycling (bool): If embed recycling, default: False.
      use_remat     (bool): If use re-materialization for Evoformers, default: False.
  """
  return functools.partial(_afex_monomer_config, 
                           model_name   =model_name,
                           msa_clusters =msa_clusters, 
                           use_dropout  =use_dropout, 
                           use_templates=use_templates,
                           use_recycling=use_recycling, 
                           use_remat    =use_remat, )


def _afex_monomer_params(model_name: str) -> TAFParams:
  # Load presets.
  assert model_name in _af_config.MODEL_PRESETS.get('monomer')
  with open(os.path.join(AF_PARAMS_DIR, f"params_{model_name}.npz"), 'rb') as f:
    params = np.load(io.BytesIO(f.read()), allow_pickle=False)
  params = _af_utils.flat_params_to_haiku(params=params)
  return params
def AFEXMonomerParams(model_name: str) -> typing.Callable[[], TAFParams]:
  r"""The AlphaFold monomer model parameters.
  
    Args:
      model_name (str): The name of the AF-monomer model.
  """
  return functools.partial(_afex_monomer_params, model_name=model_name)