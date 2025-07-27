"""Containers for the AFCheckpt modules."""
# Authors: Zilin Song

__all__ = ['InferForCheckpt', 
           'InferFromStructmod',
           'AFCHECKPT_MODULES', ]

import typing
import functools

import haiku as hk
import jax
import jax.numpy as jnp

from alphafold.model import (modules          as AF_MODULES,
                             modules_multimer as AF_MODULES_MULTIMER, 
                             folding_multimer as AF_FOLDING_MULTIMER, )

import aftools.utils.af as _u_af


PARAMS_PREFIX = {'InferForCheckpt'    : 'alphafold/alphafold_iteration', 
                 'InferFromEvoformer' : 'infer_from_evoformer',
                 'InferFromStructmod' : 'infer_from_structmod', }
                   

# InferForCheckpt. .................................................................................
def InferForCheckpt(AFMultimerConfig: typing.Callable[[], _u_af.TAFConfig],
                    AFMultimerParams: typing.Callable[[], _u_af.TAFParams], 
                    evoformer_checkpt: bool, 
                    structmod_checkpt: bool, 
                    ) -> tuple[typing.Type[hk.Module], _u_af.TAFConfig, _u_af.TAFParams]:
  """The hk.Module, the config, and the params for creating the AFMultimer checkpoints.
  
    Args:
      AFMultimerConfig  (Callable[[], TAFConfig]): The AlphaFold-Multimer model configurations.
      AFMultimerParams  (Callable[[], TAFParams]): The AlphaFold-Multimer model parameters.
      evoformer_checkpt (bool): If record the evoformer checkpoints.
      structmod_checkpt (bool): If record the structmod checkpoints.
  """
  # Config.
  config = AFMultimerConfig()
  assert config.model.global_config.multimer_mode == True
  if evoformer_checkpt or structmod_checkpt: 
    config.update_from_flattened_dict({'model.global_config.aftools': {}, })
    config.update_from_flattened_dict({
      'model.global_config.aftools.afcheckpt_is_last_recycling': False, 
      'model.global_config.aftools.afcheckpt_evoformer_checkpt': evoformer_checkpt, 
      'model.global_config.aftools.afcheckpt_structmod_checkpt': structmod_checkpt, })
  # Params. 
  params = _u_af._afmultimer_params_add_prefix(params=AFMultimerParams(), 
                                               prefix=PARAMS_PREFIX['InferForCheckpt'], )
  
  return AF_MODULES_MULTIMER.AlphaFold, config.model, params


# InferFromEvoformer. ..............................................................................
class _Evoformer(hk.Module):
  """This is a wrapper for `EvoformerIterations` such that the parameters can be correctly mapped.
    Implements the `Evoformer` corresponding to the `EmbeddingAndEvoformer` in the AF code.
    The config should be TAFConfig.model.embeddings_and_evoformer
  """
  def __init__(self, config: _u_af.TAFConfig, global_config: _u_af.TAFConfig):
    hk.Module.__init__(self, name='evoformer')
    self._config = config
    self._global_config = global_config
  
  def __call__(self, 
               input: _u_af.TAFFeatures, 
               masks: _u_af.TAFFeatures, 
               is_training: bool, 
               ) -> _u_af.TAFResults:
    """Infer the Evoformer stack. Return the inference results."""
    c, gc = self._config, self._global_config

    with _u_af.AF_UTILS.bfloat16_context(): # Convert params to bfloat16 if requested
      # data format to jnp.bfloat16, to save memory.
      if gc.bfloat16 == True:
        for k in input.keys(): input[k] = input[k].astype(jnp.bfloat16)
        for k in masks.keys(): masks[k] = masks[k].astype(jnp.bfloat16)

      # modules: evoformer stack.
      evoformer_iteration = AF_MODULES.EvoformerIteration(c.evoformer, gc, is_extra_msa=False)
      def evoformer_fn(x): 
        return evoformer_iteration(activations=x, masks=masks, is_training=is_training)
      evoformer_fn = hk.remat(evoformer_fn) if gc.use_remat == True else evoformer_fn
      evoformer_stack = AF_MODULES.layer_stack.layer_stack(c.evoformer_num_block)(evoformer_fn)

      # evoformer stack inference.
      ret_evoformer = evoformer_stack(input)

      msa_act, pair_act  = ret_evoformer['msa'], ret_evoformer['pair']

      # modules: single_activation
      single_act = AF_MODULES.common_modules.Linear(c.seq_channel, name='single_activations', 
                                                    )(msa_act[0])
      ret = {'single': single_act, 'pair': pair_act}

      # Maybe data format back to jnp.float32.
      if gc.bfloat16 == True: 
        for k in ret.keys(): ret[k] = ret[k].astype(jnp.float32)

    return ret


class _InferFromEvoformer(hk.Module):
  """The AFCheckpt inference module from the `evoformer` checkpoint through the evoformer stack, the
    structure module, the predicted LDDT head, and the predicted aligned error head.
  """

  def __init__(self, config: _u_af.TAFConfig):
    """Create an AFCheckpt inference module from the `evoformer` checkpoint through the evoformer 
      stack, the structure module, the predicted LDDT head, and the predicted aligned error head.
    
      Args:
        config (TAFConfig): The AFMultimer config dict.
    """
    hk.Module.__init__(self, name=PARAMS_PREFIX['InferFromEvoformer'])
    self._config = config
  
  def __call__(self, 
               representations: _u_af.TAFFeatures, 
               batch:           _u_af.TAFFeatures, 
               is_training: bool, 
               ) -> _u_af.TAFResults:
    """Infer with the representations through the evoformer stack, the structure module, the pLDDT 
      head, and the pAE head. Return the inference results.
    """
    # configs.
    global_config = self._config.model.global_config
    evofmr_config = self._config.model.embeddings_and_evoformer
    strmod_config = self._config.model.heads.structure_module
    plddt_config  = self._config.model.heads.predicted_lddt
    pae_config    = self._config.model.heads.predicted_aligned_error
    
    # modules: StructMod and heads.
    evofmr = _Evoformer(evofmr_config, global_config)
    strmod = AF_FOLDING_MULTIMER.StructureModule (strmod_config, global_config)
    plddt  = AF_MODULES.PredictedLDDTHead        (plddt_config,  global_config)
    pae    = AF_MODULES.PredictedAlignedErrorHead(pae_config,    global_config)

    # Unpack representations.
    evofmr_input = {'msa': representations['msa'],       'pair': representations['pair']}
    evofmr_masks = {'msa': representations['masks_msa'], 'pair': representations['masks_pair']}
    # evoformer stack inference.
    ret_evofmr = evofmr(input=evofmr_input, masks=evofmr_masks, is_training=is_training)
    # structure module inference.
    ret_strmod = strmod(representations=ret_evofmr, batch=batch, is_training=is_training)
    ret_evofmr['structure_module'] = ret_strmod.pop('act')  # For pLDDT head.
    # pLDDT head inference.
    ret_plddt = plddt(representations=ret_evofmr, batch=batch, is_training=is_training)
    # pAE head inference.
    ret_pae = pae(representations=ret_evofmr, batch=batch, is_training=is_training)
    ret_pae['asym_id'] = batch['asym_id']  # For ipTM computation.

    return {'evoformer_stack':         ret_evofmr, 
            'structure_module':        ret_strmod, 
            'predicted_lddt':          ret_plddt, 
            'predicted_aligned_error': ret_pae, }


def InferFromEvoformer(AFMultimerConfig: typing.Callable[[], _u_af.TAFConfig],
                       AFMultimerParams: typing.Callable[[], _u_af.TAFParams], 
                       ) -> tuple[typing.Type[hk.Module], _u_af.TAFConfig, _u_af.TAFParams]:
  """The hk.Module, the config, and the params for AFCheckpt inference from the structmod checkpoint
    without any adaptation.
  
    Args:
      AFMultimerConfig (Callable[[], TAFConfig]): The AlphaFold-Multimer model configurations.
      AFMultimerParams (Callable[[], TAFParams]): The AlphaFold-Multimer model parameters.
  """
  # Config.
  config = AFMultimerConfig()
  assert config.model.global_config.multimer_mode == True
  # Params. 
  params = _u_af._afmultimer_params_add_prefix(params=AFMultimerParams(), 
                                               prefix=PARAMS_PREFIX['InferFromEvoformer'], )
  return _InferFromEvoformer, config, params


# InferFromStructmod. ..............................................................................
class _InferFromStructmod(hk.Module):
  """The AFCheckpt inference module from the `structmod` checkpoint through the structure module, 
    the predicted LDDT head, and the predicted aligned error head.
  """

  def __init__(self, config: _u_af.TAFConfig):
    """Create an AFCheckpt inference module from the structmod checkpoint through the structure 
      module, the predicted LDDT head, and the predicted aligned error head.
    
      Args:
        config (TAFConfig): The AFMultimer config dict.
    """
    hk.Module.__init__(self, name=PARAMS_PREFIX['InferFromStructmod'])
    self._config = config

  def __call__(self, 
               representations: _u_af.TAFFeatures, 
               batch:           _u_af.TAFFeatures, 
               is_training:     bool, 
               ) -> _u_af.TAFResults:
    """Infer with the representations through the structure module, the pLDDT head, and the pAE 
      head. Return the inference results.
    """
    # configs.
    global_config = self._config.model.global_config
    strmod_config = self._config.model.heads.structure_module
    plddt_config  = self._config.model.heads.predicted_lddt
    pae_config    = self._config.model.heads.predicted_aligned_error
    
    # modules.
    strmod = AF_FOLDING_MULTIMER.StructureModule (strmod_config, global_config)
    plddt  = AF_MODULES.PredictedLDDTHead        (plddt_config,  global_config)
    pae    = AF_MODULES.PredictedAlignedErrorHead(pae_config,    global_config)

    # structure module inference.
    ret_strmod = strmod(representations=representations, batch=batch, is_training=is_training)
    representations['structure_module'] = ret_strmod.pop('act')  # For pLDDT head.
    # pLDDT head inference.
    ret_plddt = plddt(representations=representations, batch=batch, is_training=is_training)
    # pAE head inference.
    ret_pae = pae(representations=representations, batch=batch, is_training=is_training)
    ret_pae['asym_id'] = batch['asym_id']  # For ipTM computation.

    return {'structure_module':        ret_strmod, 
            'predicted_lddt':          ret_plddt, 
            'predicted_aligned_error': ret_pae, }


def InferFromStructmod(AFMultimerConfig: typing.Callable[[], _u_af.TAFConfig],
                       AFMultimerParams: typing.Callable[[], _u_af.TAFParams], 
                       ) -> tuple[typing.Type[hk.Module], _u_af.TAFConfig, _u_af.TAFParams]:
  """The hk.Module, the config, and the params for AFCheckpt inference from the structmod checkpoint
    without any adaptation.
  
    Args:
      AFMultimerConfig (Callable[[], TAFConfig]): The AlphaFold-Multimer model configurations.
      AFMultimerParams (Callable[[], TAFParams]): The AlphaFold-Multimer model parameters.
  """
  # Config.
  config = AFMultimerConfig()
  assert config.model.global_config.multimer_mode == True
  # Params. 
  params = _u_af._afmultimer_params_add_prefix(params=AFMultimerParams(), 
                                               prefix=PARAMS_PREFIX['InferFromStructmod'], )
  return _InferFromStructmod, config, params


# General Checkpt utils. ...........................................................................
AFCHECKPT_MODULES: dict[
  str, typing.Callable[ [typing.Callable[[], _u_af.TAFConfig], 
                         typing.Callable[[], _u_af.TAFParams], ], 
                         tuple[hk.Module, _u_af.TAFConfig, _u_af.TAFParams] ]] = {
  'evoformer': InferFromEvoformer, 
  'structmod': InferFromStructmod, }
"""The AFCheckpt modules."""


# AF confidence modules. ...........................................................................
def _plddt_bin_centers(num_bins: int) -> jnp.ndarray:
  """Compute the `bin_centers` for pLDDT score calculation."""
  # From: alphafold/common/confidence.py, line 31-33.
  bin_width = 1./float(num_bins)
  bin_centers = jnp.arange(start=.5*bin_width, stop=1., step=bin_width)
  return bin_centers
def _plddt(logits: jnp.ndarray, bin_centers: jnp.ndarray) -> jnp.ndarray:
  """JAX-differentiable implementation of the pLDDT scores."""
  ## From: alphafold/common/confidence.py, line 34-36.
  plddt = jnp.sum(jax.nn.softmax(logits, axis=-1)*bin_centers[None, :], axis=-1) # [Nres, ]
  return plddt


def _ptm_bin_wise_tm(num_res: int, num_bins: int, max_error_bin: float) -> jnp.ndarray:
  """Compute the `tm_bin_wise` for pTM score calculation."""
  # Compute `d0`.
  ## From: alphafold/common/confidence.py, line 142-147.
  num_res = int(num_res) if num_res >= 19. else 19
  d0 = 1.24 * (num_res - 15) ** (1./3.) - 1.8
  # Compute `bin_centers`.
  ## From: alphafold/model/modules.py, line 1163-1165.
  breaks = jnp.linspace(0., float(max_error_bin), num_bins-1)
  step = breaks[1] - breaks[0]
  ## From: alphafold/common/confidence.py, line 50-54.
  bin_centers = jnp.asarray(breaks + step / 2.)
  bin_centers = jnp.concatenate((bin_centers, jnp.asarray([bin_centers[-1] + step])), axis=0)
  # Compute `bin_wise_tm`, i.e., `tm_per_bin` in AF source code.
  ## From: alphafold/common/confidence.py, line 152-153.
  bin_wise_tm = 1. / (1. + jnp.square(bin_centers) / jnp.square(d0))
  return bin_wise_tm
def _ptm(logits: jnp.ndarray, bin_wise_tm: jnp.ndarray) -> jnp.ndarray:
  """JAX-differentiable implementation of the pTM scores."""
  ## From: alphafold/common/confidence.py, line 149-155. Compute the `predicted_tm_term`.
  ptm = jnp.sum(jax.nn.softmax(logits, axis=-1)*bin_wise_tm, axis=-1) # [Nres, Nres]
  return jnp.mean(ptm, axis=-1) # [Nres, ]


def _iptm_chain_mask(asym_id: jnp.ndarray) -> jnp.ndarray:
  """Compute the `chain_mask` for ipTM score calculation."""
  ## From: alphafold/common/confidence.py, line 157-159.
  return asym_id[:, None] != asym_id[None, :]
def _iptm(logits: jnp.ndarray, bin_wise_tm: jnp.ndarray, chain_mask: jnp.ndarray) -> jnp.ndarray:
  """JAX-differentiable implementation of the ipTM scores."""
  ## From: alphafold/common/confidence.py, line 149-155. Compute the `predicted_tm_term`.
  iptm = jnp.sum(jax.nn.softmax(logits, axis=-1)*bin_wise_tm, axis=-1)
  ## From: alphafold/common/confidence.py, line 165-166. Compute the `normed_residue_mask`.
  norm_chain_mask = chain_mask / (jnp.sum(chain_mask, axis=-1, keepdims=True))
  ## From: alphafold/common/confidence.py, line 167. Compute the `per_alignment`.
  return jnp.sum(iptm*norm_chain_mask, axis=-1)


def _afmultimer_confidence_impl(af_results: _u_af.TAFResults, 
                                plddt_bin_centers: jnp.ndarray, 
                                ptm_bin_wise_tm:   jnp.ndarray, 
                                iptm_chain_mask:   jnp.ndarray, 
                                ) -> _u_af.TAFResults:
  """The AFMultimer confidence impl."""
  # Unpack pLDDT and pAE logits.
  plddt_logits = af_results['predicted_lddt']['logits']
  ptm_logits   = af_results['predicted_aligned_error']['logits']
  # To scores.  
  plddt = _plddt(logits=plddt_logits, bin_centers=plddt_bin_centers)
  ptm   =  _ptm (logits=ptm_logits, bin_wise_tm=ptm_bin_wise_tm)
  iptm  = _iptm (logits=ptm_logits, bin_wise_tm=ptm_bin_wise_tm, chain_mask=iptm_chain_mask)
  return {'plddt': plddt,
          'ptm':   ptm, 
          'iptm':  iptm, 
          'confidence': {'plddt': jax.lax.stop_gradient(jnp.mean(plddt)), 
                         'ptm':   jax.lax.stop_gradient(jnp.max (ptm  )), 
                         'iptm':  jax.lax.stop_gradient(jnp.max (iptm )), 
                         'rank':  jax.lax.stop_gradient(.8*jnp.max(iptm)+.2*jnp.max(ptm))}, }
def AFMultimerConfidenceHead(plddt_num_bins:    int, 
                             ptm_num_res:       int, 
                             ptm_num_bins:      int, 
                             ptm_max_error_bin: float, 
                             iptm_asym_id:      jnp.ndarray, 
                             ) -> typing.Callable[[_u_af.TAFResults], _u_af.TAFResults]:
  """The AFMultimer confidence head.
  
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
          `results`: The AFMultimer prediction results.
        Return the confidence scores:
          `plddt`: The  pLDDT scores;
          `ptm`:   The  pTM   scores, AF takes the max value.
          `iptm`:  The ipTM   scores, AF takes the max value.
  """
  plddt_bin_centers = _plddt_bin_centers(num_bins=plddt_num_bins)
  ptm_bin_wise_tm = _ptm_bin_wise_tm(num_res=ptm_num_res, 
                                     num_bins=ptm_num_bins, 
                                     max_error_bin=ptm_max_error_bin, )
  iptm_chain_mask = _iptm_chain_mask(asym_id=iptm_asym_id)
  return jax.jit(functools.partial(_afmultimer_confidence_impl, 
                                   plddt_bin_centers=plddt_bin_centers, 
                                   ptm_bin_wise_tm=ptm_bin_wise_tm, 
                                   iptm_chain_mask=iptm_chain_mask, ))