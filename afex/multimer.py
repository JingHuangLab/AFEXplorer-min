r"""AFEXplorer multimer.

NOTE: The AFEX multimer implementation requires modifications to the AlphaFold-Multimer source code
      See: ./alphafold/model/modules_multimer.py, line 641.
"""
# Wrapper for AFEX multimer model.
# Zilin Song


import jax
import jax.numpy as jnp
import haiku     as hk

import alphafold.model.model            as _af_model
import alphafold.model.modules_multimer as _af_modules

import afex.utils as _u


class AFEXM(_af_model.RunModel):
  r"""Container for AFEX multimer model."""

  def __init__(self, config: _u.TAFConfig, params: _u.TAFParams):
    r"""Create a container for AFEX multimer model.
    
      Args:
        config (TAFConfig): AF multimer configurations, from `afex.utils.AFEXMultimerConfig()`.
        params (TAFParams): AF multimer parameters,     from `afex.utils.AFEXMultimerParams()`.
    """
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode # Always True -> multimer only.
    assert self.multimer_mode==True, r"The AFEX multimer container must be used in multimer mode."

    def _forward_fn(batch: _u.TAFFeatures):
      model = _af_modules.AlphaFold(config=self.config)
      return model(batch                 =batch, 
                   is_training           =False, 
                   return_representations=False, 
                   safe_key              =None, )

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init  = jax.jit(hk.transform(_forward_fn).init )

  def forward(self, batch: _u.TAFFeatures, afex_feat: jnp.ndarray, rand_seed: int) -> _u.TAFResults:
    r"""The forward pass.
    
      Args:
        batch     (FeatureDict): The AF features, loaded from `features.pkl`.
        afex_feat (jnp.ndarray): The AFEX features.
        rand_seed (int)        : The random seed.
      
      Returns:
        A dictionary of model outputs.
    """
    batch['afex_feat'] = afex_feat
    res = self.apply(self.params, jax.random.PRNGKey(seed=rand_seed), batch)
    return res