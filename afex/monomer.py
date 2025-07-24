# Wrapper for AFEX model runner.
# Zilin Song

"""AFEXplorer RunModel."""


import jax
import jax.numpy as jnp
import haiku     as hk

import alphafold.model.model   as _af_model
import alphafold.model.modules as _af_modules

import afex.utils as _u


class AFEX(_af_model.RunModel):
  r"""Container for AFEX-monomer model."""

  def __init__(self, config: _u.TAFConfig, params: _u.TAFParams):
    r"""Create a container for AFEX-monomer model.
    
      Args:
        config (TAFConfig): AF monomer configurations, from `afex.utils.AFEXMonomerConfig()`.
        params (TAFParams): AF monomer parameters,     from `afex.utils.AFEXMonomerParams()`.
    """
    self.config = config
    self.params = params
    self.multimer_mode = config.model.global_config.multimer_mode # Always False -> Monomer only.
    assert self.multimer_mode==False, r"The AFEX-monomer container must be used in monomer mode."

    def _forward_fn(batch: _u.TAFFeatures):
      model = _af_modules.AlphaFold(self.config.model)
      batch['msa_feat'] = batch['msa_feat'].at[:, :, :, 25:48].mul(batch['afex_feat'])
      return model(batch, 
                   is_training             =False, 
                   compute_loss            =False, 
                   ensemble_representations=False, 
                   return_representations  =False, )

    self.apply = jax.jit(hk.transform(_forward_fn).apply)
    self.init  = jax.jit(hk.transform(_forward_fn).init )
  
  def predict(self, batch: _u.TAFFeatures, afex_feat: jnp.ndarray, rand_seed: int) -> _u.TAFResults:
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