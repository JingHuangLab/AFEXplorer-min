"""AlphaFoldTools AFProtac model containers."""
# Authors: Zilin Song

__all__ = ['AFProtacInferenceModel', 'AFProtacInferenceModelConfig', ]

import typing

import numpy as np
import jax
import jax.numpy as jnp

import aftools.utils.af as _u_af
import aftools.aflora.models_multimer     as _aflora_mod
import aftools.afcheckpt.modules_multimer as _afchk_m
import aftools.afprotac.modules_multimer  as _afprotac_m


class AFProtacInferenceModelConfig: 
  """The configuration for AFProtacInferenceModel."""

  def __init__(self):
    """Create a configuration for AFProtacInferenceModel."""
    # Loss configs.
    self.centd_d0_core: float = -45.
    """The reference centroid distance between the   core region (negative to push far)."""
    self.centd_d0_pock: float =  20.
    """The reference centroid distance between the pocket region (negative to push far)."""
    self.centd_d0_grip: float = -65.
    """The reference centroid distance between the   grip region (negative to push far)."""
    self.centd_d0_link: float = 3.
    """The reference centroid distance between the linker clusters (negative to push far)."""
    self.knn_neighbors: int = 15
    """The number of neighbors for kNN density estimation."""

    ## Loss weight configs.
    ## alignment errors.
    self.w_align_core: float = 0.
    """The loss weight on the alignment error of the   core region."""
    self.w_align_pock: float = 0.
    """The loss weight on the alignment error of the pocket region."""
    self.w_align_grip: float = 0.
    """The loss weight on the alignment error of the   grip region."""
    ## centroid distances.
    self.w_centd_core: float = 5.
    """The loss weight on the centroid distance between the   core region."""
    self.w_centd_pock: float = 5.
    """The loss weight on the centroid distance between the pocket region."""
    self.w_centd_grip: float = 5.
    """The loss weight on the centroid distance between the   grip region."""
    self.w_centd_link_core: float = 5.
    """The loss weight on the centroid distance between the linker   core region."""
    self.w_centd_link_pock: float = 5.
    """The loss weight on the centroid distance between the linker pocket region."""
    self.w_centd_link_grip: float = 5.
    """The loss weight on the centroid distance between the linker   grip region."""
    ## negative likelihood.
    self.w_negll_core: float = 10.
    """The loss weight on the negative likelihood of the aligned linker   core region."""
    self.w_negll_pock: float = 10.
    """The loss weight on the negative likelihood of the aligned linker pocket region."""
    self.w_negll_grip: float = 10.
    """The loss weight on the negative likelihood of the aligned linker   grip region."""
    ## confidence scores
    self.w_confd_plddt: float = 0.
    """The loss weight on the confidence pLDDT."""
    self.w_confd_ptm:   float = 0.
    """The loss weight on the confidence pTM."""
    self.w_confd_iptm:  float = 0.
    """The loss weight on the confidence ipTM."""


class AFProtacInferenceModel:
  """Container for AlphaFold-Multimer AFProtac inference."""

  def __init__(self, 
               config: AFProtacInferenceModelConfig, 
               AFMultimerConfig:   typing.Callable[[], _u_af.TAFConfig], 
               AFMultimerParams:   typing.Callable[[], _u_af.TAFParams], 
               AFMultimerLoraFeat: typing.Callable[[], _u_af.TAFFeatures],
               mm_cor_ligase: jnp.ndarray, 
               mm_cor_target: jnp.ndarray,  
               idxes: dict[str, dict[str, dict[str, jnp.ndarray]]], 
               mm_cor_ligase_link: jnp.ndarray, 
               mm_cor_target_link: jnp.ndarray, ):
    """Create a container for AlphaFold-Multimer AFProtac inference.
    
      Args:
        config (AFProtacInferenceModelConfig):          The AFProtac inference configurations.
        AFMultimerConfig   (Callable[[], TAFConfig]):   The AlphaFold-Multimer model configurations.
        AFMultimerParams   (Callable[[], TAFParams]):   The AlphaFold-Multimer model parameters.
        AFMultimerLoraFeat (Callable[[], TAFFeatures]): The AFCheckpt LoRA features.
        mm_cor_ligase (jnp.ndarray): The coordinates of the ligase protein.
        mm_cor_target (jnp.ndarray): The coordinates of the target protein.
        idxes (dict[str, dict[str, dict[str, jnp.ndarray]]]): The AFProtac selection indexes.
        mm_cor_ligase_link (jnp.ndarray): The OpenMM linker coordinates, [Nsets, Nsamples, 3].
        mm_cor_target_link (jnp.ndarray): The OpenMM linker coordinates, [Nsets, Nsamples, 3].
    """
    # Unpack AF fixed features.
    af_lora_feat = AFMultimerLoraFeat()
    af_repr       = af_lora_feat['representations']
    af_batch      = af_lora_feat['batch']
    checkpt_from  = af_lora_feat['checkpt_from']
    lora_apply_to = af_lora_feat['lora_apply_to']
    # Graph components.
    af_model = _aflora_mod.InferFromCheckptWithLoraModel(AFMultimerConfig=AFMultimerConfig, 
                                                         AFMultimerParams=AFMultimerParams, 
                                                         checkpt_from=checkpt_from, 
                                                         lora_apply_to=lora_apply_to, )
    af_confidence_head = _afchk_m.AFMultimerConfidenceHead(
                plddt_num_bins   =af_model.config.model.heads.predicted_lddt.num_bins, 
                ptm_num_res      =int(af_batch['aatype'].shape[0]), 
                ptm_num_bins     =af_model.config.model.heads.predicted_aligned_error.num_bins, 
                ptm_max_error_bin=af_model.config.model.heads.predicted_aligned_error.max_error_bin, 
                iptm_asym_id     =np.asarray(af_batch['asym_id']), )
    afprotac_loss_head = _afprotac_m.AFProtacLossHead(mm_cor_lig=mm_cor_ligase, 
                                                      mm_cor_tar=mm_cor_target, 
                                                      idxes=idxes, 
                                                      mm_cor_lig_link=mm_cor_ligase_link, 
                                                      mm_cor_tar_link=mm_cor_target_link, 
                                                      d0_core=config.centd_d0_core,
                                                      d0_pock=config.centd_d0_pock, 
                                                      d0_grip=config.centd_d0_grip, 
                                                      d0_link=config.centd_d0_link, 
                                                      k      =config.knn_neighbors, )
    # The inference graph.
    def _forward_fn(_lora: _u_af.TAFFeatures) -> _u_af.TAFResults:
      """The forward pass."""
      # Inference.
      _res_af = af_model.forward(afmultimer_repr =af_repr, 
                                 afmultimer_batch=af_batch, 
                                 afmultimer_lora =_lora, 
                                 rseed=0, )
      _res_confidence = af_confidence_head(af_results=_res_af)
      _res_afprotac   = afprotac_loss_head(af_results=_res_af)
      # Pack misc. results.
      _res = {'af':           _res_af,
              'afconfidence': {'confidence': _res_confidence['confidence']}, 
              'afprotac':     {'centd':      _res_afprotac  ['centd'],  }, }
                              #  'structure_module': _res_afprotac['structure_module'], }, }
      # Compute losses - align.
      _loss_align_core_ligase = _res_afprotac['align_loss']['core_ligase']
      _loss_align_core_target = _res_afprotac['align_loss']['core_target']
      _loss_align_pock_ligase = _res_afprotac['align_loss']['pock_ligase']
      _loss_align_pock_target = _res_afprotac['align_loss']['pock_target']
      _loss_align_grip_ligase = _res_afprotac['align_loss']['grip_ligase']
      _loss_align_grip_target = _res_afprotac['align_loss']['grip_target']
      # Compute losses - centd.
      _loss_centd_core      = _res_afprotac['centd_loss']['core']
      _loss_centd_pock      = _res_afprotac['centd_loss']['pock']
      _loss_centd_grip      = _res_afprotac['centd_loss']['grip']
      _loss_centd_link_core = _res_afprotac['centd_loss']['link_core']  # [Nsets,]
      _loss_centd_link_pock = _res_afprotac['centd_loss']['link_pock']  # [Nsets,]
      _loss_centd_link_grip = _res_afprotac['centd_loss']['link_grip']  # [Nsets,]
      # Compute losses - negll.
      _loss_negll_core_ligase = _res_afprotac['negll_loss']['core_ligase']  # [Nsets,]
      _loss_negll_core_target = _res_afprotac['negll_loss']['core_target']  # [Nsets,]
      _loss_negll_pock_ligase = _res_afprotac['negll_loss']['pock_ligase']  # [Nsets,]
      _loss_negll_pock_target = _res_afprotac['negll_loss']['pock_target']  # [Nsets,]
      _loss_negll_grip_ligase = _res_afprotac['negll_loss']['grip_ligase']  # [Nsets,]
      _loss_negll_grip_target = _res_afprotac['negll_loss']['grip_target']  # [Nsets,]
      # Compute losses - confidence.
      _loss_confd_plddt = 1.-jnp.mean(_res_confidence[ 'plddt'])
      _loss_confd_ptm   = 1.-jnp.mean(_res_confidence[ 'ptm'  ])
      _loss_confd_iptm  = 1.-jnp.mean(_res_confidence['iptm'  ])
      # Pack raw losses.
      _res.update({'loss_raw': {'align_core_ligase': jax.lax.stop_gradient(_loss_align_core_ligase),
                                'align_core_target': jax.lax.stop_gradient(_loss_align_core_target),
                                'align_pock_ligase': jax.lax.stop_gradient(_loss_align_pock_ligase),
                                'align_pock_target': jax.lax.stop_gradient(_loss_align_pock_target),
                                'align_grip_ligase': jax.lax.stop_gradient(_loss_align_grip_ligase),
                                'align_grip_target': jax.lax.stop_gradient(_loss_align_grip_target),
                                'centd_core':      jax.lax.stop_gradient(_loss_centd_core),
                                'centd_pock':      jax.lax.stop_gradient(_loss_centd_pock),
                                'centd_grip':      jax.lax.stop_gradient(_loss_centd_grip),
                                'centd_link_core': jax.lax.stop_gradient(_loss_centd_link_core),
                                'centd_link_pock': jax.lax.stop_gradient(_loss_centd_link_pock),
                                'centd_link_grip': jax.lax.stop_gradient(_loss_centd_link_grip),
                                'negll_core_ligase': jax.lax.stop_gradient(_loss_negll_core_ligase),
                                'negll_core_target': jax.lax.stop_gradient(_loss_negll_core_target),
                                'negll_pock_ligase': jax.lax.stop_gradient(_loss_negll_pock_ligase),
                                'negll_pock_target': jax.lax.stop_gradient(_loss_negll_pock_target),
                                'negll_grip_ligase': jax.lax.stop_gradient(_loss_negll_grip_ligase),
                                'negll_grip_target': jax.lax.stop_gradient(_loss_negll_grip_target),
                                'confd_plddt': jax.lax.stop_gradient(_loss_confd_plddt),
                                'confd_ptm'  : jax.lax.stop_gradient(_loss_confd_ptm  ),
                                'confd_iptm' : jax.lax.stop_gradient(_loss_confd_iptm ), }, })
      # Compute losses - weighted.
      _loss_align = (  (_loss_align_core_ligase+_loss_align_core_target)*config.w_align_core
                     + (_loss_align_pock_ligase+_loss_align_pock_target)*config.w_align_pock
                     + (_loss_align_grip_ligase+_loss_align_grip_target)*config.w_align_grip)
      _loss_centd = (  _loss_centd_core              *config.w_centd_core
                     + _loss_centd_pock              *config.w_centd_pock
                     + _loss_centd_grip              *config.w_centd_grip
                     + jnp.sum(_loss_centd_link_core)*config.w_centd_link_core
                     + jnp.sum(_loss_centd_link_pock)*config.w_centd_link_pock
                     + jnp.sum(_loss_centd_link_grip)*config.w_centd_link_grip)
      _loss_negll = jnp.sum(  (_loss_negll_core_ligase+_loss_negll_core_target)*config.w_negll_core
                            + (_loss_negll_pock_ligase+_loss_negll_pock_target)*config.w_negll_pock
                            + (_loss_negll_grip_ligase+_loss_negll_grip_target)*config.w_negll_grip)
      _loss_confd = (  _loss_confd_plddt * config.w_confd_plddt
                     + _loss_confd_ptm   * config.w_confd_ptm
                     + _loss_confd_iptm  * config.w_confd_iptm)
      _loss_total = _loss_align + _loss_centd + _loss_negll + _loss_confd
      # Pack losses - weighted.
      _res.update({'loss_stat': {'align': jax.lax.stop_gradient(_loss_align), 
                                 'centd': jax.lax.stop_gradient(_loss_centd), 
                                 'negll': jax.lax.stop_gradient(_loss_negll), 
                                 'confd': jax.lax.stop_gradient(_loss_confd), 
                                 'total': jax.lax.stop_gradient(_loss_total), }, })
      _res.update({'loss': _loss_total})
      return _res
    self.app = _forward_fn # Do not jit: All forward functions have been JIT-ed.

  def forward(self, _lora: _u_af.TAFFeatures) -> _u_af.TAFResults:
    """Forward pass of the inference model."""
    return self.app(_lora)