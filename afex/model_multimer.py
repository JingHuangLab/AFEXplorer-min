r"""The AFEX-Multimer model.
NOTE: The AFEX-Multimer implementation requires modifications to the AF-Multimer source code:
      See: `./alphafold/model/modules_multimer.py`, line 639-641.
"""
# Authors: Zilin Song.


import copy
import typing

import jax
import jax.numpy as jnp
import haiku     as hk

import alphafold.model.config           as _af_config
import alphafold.model.modules_multimer as _af_modules

import afex.utils as _u


class AFEXMultimer:
  r"""The container for the AFEX-Multimer model."""
  
  def __init__(self, config: _u.TAFConfig, params: _u.TAFParams):
    r"""Crate a container for the AFEX-Multimer model.
    
      Args:
        config (TAFConfig): The AF-Multimer base model configurations.
        params (TAFParams): The AF-Multimer base model parameters.
    """
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode # Always True -> multimer only.
    assert self.multimer_mode==True, r"The AFEX-Multimer container must be used in `multimer` mode."

    def _forward_fn(batch: _u.TAFFeatures):
      model = _af_modules.AlphaFold(config=self.config.model)
      return model(batch                 =batch, 
                   is_training           =False, 
                   return_representations=False, 
                   safe_key              =None, )

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init  = jax.jit(hk.transform(_forward_fn).init )
  
  @property
  def nclus(self) -> int:
    return self.config.model.embeddings_and_evoformer.num_msa
  
  @property
  def ntoks(self) -> int:
    return 23 # 20 AAs + Mask + Gap + Unknown

  def forward(self, 
              feat_af:   _u.TAFFeatures, 
              feat_afex: jnp.ndarray, 
              rand_seed: int, 
              ) -> _u.TAFResults:
    r"""The forward pass.
    
      Args:
        feat_af   (FeatureDict): The AF features, loaded from `features.pkl`.
        feat_afex (jnp.ndarray): The AFEX features.
        rand_seed (int)        : The random seed for model inference, controls the MSA sampling.
      
      Returns:
        A dictionary of model outputs.
    """
    feat_af['afex_feat'] = feat_afex
    res = self.apply(self.params, jax.random.PRNGKey(seed=rand_seed), feat_af)
    return res


# AF-Multimer base model configurations.
def AFEXMultimerConfig(model_name:    str,
                       msa_clusters:  int  = 508,
                       use_dropout:   bool = False,
                       use_template:  bool = False,
                       use_recycling: bool = False,
                       use_remat:     bool = True,
                       ) -> _u.TAFConfig:
  """The AF-Multimer base model configurations.
  
    Args:
      model_name    (str):  The name of the AF-Multimer base model.
      msa_clusters  (int):  The number of MSA clusters, default: 508 (plus 4 templates).
      use_dropout   (bool): If use dropout during forwarding, default: False.
      use_templates (bool): If embed templates, default: False.
      use_recycling (bool): If embed recycling, default: False.
      use_remat     (bool): If use re-materialization for Evoformers, default: False.
  """
  # Load presets.
  assert model_name in _af_config.MODEL_PRESETS.get('multimer')
  config = copy.deepcopy(_af_config.CONFIG_MULTIMER)
  config.update_from_flattened_dict(_af_config.CONFIG_DIFFS[model_name])
  # Turn on multimer_mode.
  config.model.global_config.multimer_mode = True
  # Single ensemble.
  config.model.num_ensemble_eval = 1
  # Maybe reduce `num_msa_clusters` for efficient memory scaling.
  config.model.embeddings_and_evoformer.num_msa = msa_clusters
  # If use dropout.
  config.model.global_config.deterministic = not use_dropout
  # If use template.
  config.model.embeddings_and_evoformer.template.enabled = use_template
  # If use recycling.
  config.model.num_recycle                               = 0 if not use_recycling else 20
  config.model.resample_msa_in_recycling                 = use_recycling
  config.model.embeddings_and_evoformer.recycle_pos      = use_recycling
  config.model.embeddings_and_evoformer.recycle_features = use_recycling
  # If use re-materialization (reverse-mode autograd where necessary).
  config.model.global_config.use_remat = use_remat
  return config


# AF-Multimer base model parameters.
def AFEXMultimerParams(model_name: str) -> _u.TAFParams:
  r"""The AF-Multimer base model configurations.
  
    Args:
      model_name (str): The name of the AF-Multimer base model.
  """
  assert model_name in _af_config.MODEL_PRESETS.get('multimer')
  return _u.load_params(model_name=model_name)