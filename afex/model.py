r"""The AFEX model."""
# Authors: Zilin Song.


import copy
import typing
import functools

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
  config.data.common.resample_msa_in_recycling           = use_recycling
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


# AF confidence outputs.
def _plddt_bin_center(num_bins: int) -> jnp.ndarray:
  r"""Returns the bin centers for pLDDT computation.
  
    Args:
      num_bins (int): `config.model.heads.predicted_lddt.num_bins`.
  """
  # From: alphafold/common/confidence.py, line 31-36.
  bin_width = 1./float(num_bins)
  bin_centers = jnp.arange(start=.5*bin_width, stop=1., step=bin_width)
  return bin_centers
def _plddt(logits: jnp.ndarray, bin_centers: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of the AF pLDDT prediction head.

    Args:
      logits      (jnp.ndarray): `AFResults['predicted_lddt']['logits']`.
      bin_centers (jnp.ndarray): The bin centers.
    
    Returns:
      pLDDT (jnp.ndarray): the residue-wise pLDDT.
  """
  # From: alphafold/common/confidence.py, line 34-36.
  return jnp.sum(jax.nn.softmax(logits, axis=-1) * bin_centers[None, :], axis=-1) # [Nres, ]


def _ptm_bin_tm(num_res: int, num_bins: int, max_error_bin: float) -> jnp.ndarray:
  r"""Returns the bin-wise TM for pTM and ipTM computation.
  
    Args:
      num_res       (int):   `af_batch['aatype'].shape[0]`.
      num_bins      (int):   `af_config.model.heads.predicted_aligned_error.num_bins`.
      max_error_bin (float): `af_config.model.heads.predicted_aligned_error.max_error_bin`.
  """
  # Compute `d0`.
  ## From: alphafold/common/confidence.py, line 142-147.
  num_res = int(num_res) if num_res >= 19. else 19
  d0      = 1.24 * (num_res - 15) ** (1./3.) - 1.8
  # Compute `bin_centers`.
  ## From: alphafold/model/modules.py, line 1160-1162.
  breaks = jnp.linspace(0., float(max_error_bin), num_bins-1)
  ## From: alphafold/common/confidence.py, line 50-54.
  step = breaks[1] - breaks[0]
  bin_centers = jnp.asarray(breaks + step / 2.)
  bin_centers = jnp.concatenate((bin_centers, jnp.asarray([bin_centers[-1] + step])), axis=0)
  # From: alphafold/common/confidence.py, line 152-153.
  bin_tm = 1. / (1. + jnp.square(bin_centers) / jnp.square(d0))
  return bin_tm
def _ptm(logits: jnp.ndarray, bin_tm: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of the AF pTM prediction head.
  
    Args:
      logits (jnp.ndarray): `AFResults['predicted_aligned_error']['logits']`.
      bin_tm (jnp.ndarray): The bin-wise TM score.

    Returns: 
      pTM (jnp.ndarray): the residue-wise pTM score.
  """
  # From: alphafold/common/confidence.py, line 149-155.
  return jnp.mean(jnp.sum(jax.nn.softmax(logits, axis=-1)*bin_tm, axis=-1), axis=-1)

def _af_confidence_impl(res: _u.TAFResults, 
                        plddt_bin_centers: jnp.ndarray,
                        ptm_bin_tm: jnp.ndarray, 
                        ) -> _u.TAFResults:
  r"""The AF confidence heads impl."""
  # unpack logits.
  logits_plddt = res['predicted_lddt'         ]['logits']
  logits_ptm   = res['predicted_aligned_error']['logits']
  # to scores.
  plddt = _plddt(logits=logits_plddt, bin_centers=plddt_bin_centers)
  ptm   = _ptm  (logits=logits_ptm, bin_tm=ptm_bin_tm)
  return dict( plddt= plddt, 
               ptm  = ptm  , )
def AFConfidenceHead(plddt_num_bins: int, 
                     ptm_num_res:    int, 
                     ptm_num_bins:   int, 
                     ptm_max_error_bin: float, 
                     ) -> typing.Callable[[_u.TAFResults], _u.TAFResults]:
  r"""The AF confidence head.
  
    Args:
      plddt_num_bins    (int):   From `af_config.model.heads.predicted_lddt.num_bins`.
      ptm_num_res       (int):   From `af_batch['aatype'].shape[0]`.
      ptm_num_bins      (int):   From `af_config.model.heads.predicted_aligned_error.num_bins`.
      ptm_max_error_bin (float): From `af_config.model.heads.predicted_aligned_error.max_error_bin`.
      iptm_asym_id      (jnp.ndarray): From `af_batch['asym_id']`.
    
    Returns:
      AFConfidenceHead (typing.Callable[[TAFResults], TAFResults]):
        The confidence impl. 
        Takes the AF prediction:
          `res`: The AF prediction results.
        Return the confidence scores as dictionary with the follwing keys.
          `plddt`: The  pLDDT scores;
          `ptm`:   The  pTM   scores, AF takes the max value.
          `iptm`:  The ipTM   scores, AF takes the max value.
  """
  plddt_bin_centers = _plddt_bin_center(num_bins=plddt_num_bins)
  ptm_bin_tm        = _ptm_bin_tm(num_res      =ptm_num_res, 
                                  num_bins     =ptm_num_bins, 
                                  max_error_bin=ptm_max_error_bin, )
  return functools.partial(_af_confidence_impl, 
                           plddt_bin_centers=plddt_bin_centers,
                           ptm_bin_tm=ptm_bin_tm, )