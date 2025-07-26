r"""The AFEX model."""
# Authors: Zilin Song.


import copy
import typing

import jax
import jax.numpy as jnp
import haiku     as hk

import alphafold.model.config  as _af_config
import alphafold.model.modules as _af_modules

import afex.utils as _u


class AFEX:
  r"""The container for the AFEX model."""

  def __init__(self, config: _u.TAFConfig, params: _u.TAFParams):
    r"""Create a container for the AFEX model.
    
      Args:
        config (TAFConfig): The AF base model configurations.
        params (TAFParams): The AF base model parameters.
    """
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode # Always False -> monomer only.
    assert self.multimer_mode==False, r"The AFEX container must be used in `monomer` mode."

    def _forward_fn(batch: _u.TAFFeatures):
      model = _af_modules.AlphaFold(self.config.model)
      batch['msa_feat'] = batch['msa_feat'].at[:, :, :, 25:48].mul(batch['afex_feat'])
      return model(batch                   =batch, 
                   is_training             =False, 
                   compute_loss            =False, 
                   ensemble_representations=False, 
                   return_representations  =False, )

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init  = jax.jit(hk.transform(_forward_fn).init )
  
  def forward(self, 
              feat_af:   _u.TAFFeatures, 
              feat_afex: jnp.ndarray, 
              rand_seed: int, 
              ) -> _u.TAFResults:
    r"""The forward pass.
    
      Args:
        feat_af   (FeatureDict): The AF features, loaded from `features.pkl`.
        feat_afex (jnp.ndarray): The AFEX features.
        rand_seed (int)        : The random seed.
      
      Returns:
        A dictionary of model outputs.
    """
    feat_af['afex_feat'] = feat_afex
    res = self.apply(self.params, jax.random.PRNGKey(seed=rand_seed), feat_af)
    return res
  

# AF base model configurations.
def AFEXConfig(model_name:    str, 
               msa_clusters:  int  = 512, 
               use_dropout:   bool = False, 
               use_templates: bool = False, 
               use_recycling: bool = False, 
               use_remat:     bool = True,
               ) -> _u.TAFConfig:
  # Load presets.
  assert model_name in _af_config.MODEL_PRESETS.get('monomer')
  config = copy.deepcopy(_af_config.CONFIG)
  config.update_from_flattened_dict(_af_config.CONFIG_DIFFS[model_name])
  # NOTE: AF monomer preprocesses features in alphafold.model.RunModel().process_features(), 
  #       so that config.data should be modified as well.
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


# AF base model parameters.
def AFEXParams(model_name: str) -> typing.Callable[[], _u.TAFParams]:
  r"""The AF base model parameters.
  
    Args:
      model_name (str): The name of the AF base model.
  """
  assert model_name in _af_config.MODEL_PRESETS.get('monomer')
  return _u.load_params(model_name=model_name)