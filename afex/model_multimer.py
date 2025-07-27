r"""The AFEX-Multimer model.
NOTE: The AFEX-Multimer implementation requires modifications to the AF-Multimer source code:
      See: `./alphafold/model/modules_multimer.py`, line 639-641.
"""
# Authors: Zilin Song.


import copy
import typing
import functools

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


# AF-Multimer confidence outputs.
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
  r"""JAX-differentiable implementation of the AF-Multimer pLDDT prediction head.

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
  r"""JAX-differentiable implementation of the AF-Multimer pTM prediction head.
  
    Args:
      logits (jnp.ndarray): `AFResults['predicted_aligned_error']['logits']`.
      bin_tm (jnp.ndarray): The bin-wise TM score.

    Returns: 
      pTM (jnp.ndarray): the residue-wise pTM score.
  """
  # From: alphafold/common/confidence.py, line 149-155.
  return jnp.mean(jnp.sum(jax.nn.softmax(logits, axis=-1)*bin_tm, axis=-1), axis=-1)


def _iptm_pair_mask(asym_id: jnp.ndarray) -> jnp.ndarray:
  r"""Returns the pair masks for ipTM computation.
  
    Args:
      asym_id (jnp.ndarray): The chain identifier.
  """
  return asym_id[:, None] != asym_id[None, :]
def _iptm(logits: jnp.ndarray, bin_tm: jnp.ndarray, pair_mask: jnp.ndarray) -> jnp.ndarray:
  r"""JAX-differentiable implementation of the AF-Multimer ipTM prediction head.
  
    Args:
      bin_tm  (jnp.ndarray): The bin-wise TM score.
      pair_mask (jnp.ndarray): The pair mask.
      
    Returns:
      ipTM (Callable[[jnp.ndarray], jnp.ndarray]): The residue-wise interface pTM score.
  """
  # From: alphafold/common/confidence.py, line 149-159.
  iptm = jnp.sum(jax.nn.softmax(logits, axis=-1) * bin_tm, axis=-1)
  # From: alphafold/common/confidence.py, line 165-167.
  pair_mask = pair_mask / jnp.sum(pair_mask, axis=-1, keepdims=True)
  return jnp.sum(iptm * pair_mask, axis=-1)

def _af_multimer_confidence_impl(res: _u.TAFResults, 
                                 plddt_bin_centers: jnp.ndarray,
                                 ptm_bin_tm: jnp.ndarray, 
                                 iptm_pair_mask: jnp.ndarray, 
                                 ) -> _u.TAFResults:
  r"""The AF-Multimer confidence heads impl."""
  # unpack logits.
  logits_plddt = res['predicted_lddt'         ]['logits']
  logits_ptm   = res['predicted_aligned_error']['logits']
  # to scores.
  plddt =  _plddt(logits=logits_plddt, bin_centers=plddt_bin_centers)
  ptm   =  _ptm  (logits=logits_ptm, bin_tm=ptm_bin_tm)
  iptm  = _iptm  (logits=logits_ptm, bin_tm=ptm_bin_tm, pair_mask=iptm_pair_mask)
  return dict( plddt= plddt, 
               ptm  = ptm  ,
              iptm  =iptm  )
def AFMultimerConfidenceHead( plddt_num_bins: int, 
                              ptm_num_res:    int, 
                              ptm_num_bins:   int, 
                              ptm_max_error_bin: float, 
                             iptm_asym_id:       jnp.ndarray, 
                             ) -> typing.Callable[[_u.TAFResults], _u.TAFResults]:
  r"""The AF-Multimer confidence head.
  
    Args:
      plddt_num_bins    (int):   From `af_config.model.heads.predicted_lddt.num_bins`.
      ptm_num_res       (int):   From `af_batch['aatype'].shape[0]`.
      ptm_num_bins      (int):   From `af_config.model.heads.predicted_aligned_error.num_bins`.
      ptm_max_error_bin (float): From `af_config.model.heads.predicted_aligned_error.max_error_bin`.
      iptm_asym_id      (jnp.ndarray): From `af_batch['asym_id']`.
    
    Returns:
      AFMultimerConfidenceHead (typing.Callable[[TAFResults], TAFResults]):
        The confidence impl. 
        Takes the AF prediction:
          `res`: The AFMultimer prediction results.
        Return the confidence scores as dictionary with the follwing keys.
          `plddt`: The  pLDDT scores;
          `ptm`:   The  pTM   scores, AF takes the max value.
          `iptm`:  The ipTM   scores, AF takes the max value.
  """
  plddt_bin_centers = _plddt_bin_center(num_bins=plddt_num_bins)
  ptm_bin_tm        = _ptm_bin_tm(num_res      =ptm_num_res, 
                                  num_bins     =ptm_num_bins, 
                                  max_error_bin=ptm_max_error_bin, )
  iptm_pair_mask    = _iptm_pair_mask(asym_id=iptm_asym_id)
  return functools.partial(_af_multimer_confidence_impl, 
                           plddt_bin_centers=plddt_bin_centers,
                           ptm_bin_tm=ptm_bin_tm, 
                           iptm_pair_mask=iptm_pair_mask, )