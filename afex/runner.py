r"""The AFEX model runner."""
# Authors: Zilin Song.


import os
import typing
import logging

import optax
import jax
import jax.numpy as jnp
import     numpy as  np

import alphafold.model.features as _af_feat
import alphafold.common.protein as _af_prot

import afex.model as _m
import afex.utils as _u


class AFEXRunner:
  r"""The runner for the AFEX model."""

  def __init__(self, 
               work_dir: str, 
               feat_dir: str, 
               afex_config: _u.TAFConfig, 
               afex_params: _u.TAFParams, ):
    r"""Create the runner for the AFEX-Multimer model.
    
      Args:
        work_dir (str):     The directory to which the results output.
        feat_dir (str):     The directory to the `features.pkl`.
        config (TAFConfig): The AF-Multimer base model configurations.
        params (TAFParams): The AF-Multimer base model parameters.
    """
    # private presets.
    self._whoami = 'AFEX-Multimer'
    self._logger = logging.getLogger(name=self._whoami)
    self._logger.setLevel('INFO')
    # directories.
    self.work_dir = str(work_dir)
    self.feat_dir = str(feat_dir)
    self.pdbs_dir = os.path.join(self.work_dir, 'afex_pdbs')
    self.afex_dir = os.path.join(self.work_dir, 'afex_checkpoints')
    # components.
    self.afex = _m.AFEX(config=afex_config, params=afex_params)

  @property
  def whoami(self) -> str: return self._whoami
  @property
  def logger(self) -> logging.Logger: return self._logger

  def execute(self, 
              colvar_fn:    typing.Callable[[jnp.ndarray], jnp.ndarray], 
              feat_afex:    jnp.ndarray = None, 
              optimizer:    optax.GradientTransformation = optax.adam(learning_rate=.01),
              optim_nsteps: int = 100, 
              loss_weights: tuple[float, float, float, float] = (1., 1., 1.), ):
    r"""Execute the AFEX-Multimer run.
    
      Args:
        colvar_fn (Callable[[jnp.ndarray], jnp.ndarray]):
          The function for computing the CV loss, must return a scalar tensor.
        feat_afex (jnp.ndarray): 
          The AFEX feature vector from which the optimization begins, default: None (cold start).
        optimizer (Callable):
          The optimizer, default: `optax.adam(learning_rate=.01)`.
        optim_nsteps (int):
          The number of optimization steps to execute, default: 100.
        loss_weights (tuple[float, float, float, float]):
          The weighting factor for the losses: `colvar`, `pLDDT`, `pTM`.
    """
    # prepare io.
    if not os.path.exists(self.work_dir):
      os.makedirs(self.work_dir)
      self.logger.info(f"Created directory: {self.work_dir}.")

    if not os.path.exists(self.pdbs_dir):
      os.makedirs(self.pdbs_dir)
      self.logger.info(f"Created directory: {self.pdbs_dir}.")

    if not os.path.exists(self.afex_dir):
      os.makedirs(self.afex_dir)
      self.logger.info(f"Created directory: {self.afex_dir}.")

    # prepare features - AF.
    feat_af: _u.TAFFeatures = np.load(self.feat_dir, allow_pickle=True)
    self.logger.info(f"Loaded AF features: {feat_af.keys()}.")
    feat_af = _af_feat.np_example_to_features(np_example=feat_af, config=self.afex.config, random_seed=0)
    self.logger.info(f"Processed AF features: {feat_af.keys()}.")
    
    # unpack loss weights.
    _w_colvar, _w_plddt, _w_ptm = loss_weights

    # create the confidence head for loss bp.
    confidence_head = _m.AFConfidenceHead(
      plddt_num_bins   =self.afex.config.model.heads.predicted_lddt.num_bins, 
      ptm_num_res      =int(feat_af['aatype'].shape[0]), 
      ptm_num_bins     =self.afex.config.model.heads.predicted_aligned_error.num_bins, 
      ptm_max_error_bin=self.afex.config.model.heads.predicted_aligned_error.max_error_bin, )
    
    def _forward(_feat_afex: optax.Params) -> tuple[jnp.ndarray, dict]:
      _res = self.afex.forward(feat_af=feat_af, feat_afex=_feat_afex, rand_seed=0)
      # loss - colvar.
      _loss_colvar = colvar_fn(_res['structure_module']['final_atom_positions'])
      # loss - confidence.
      _res_confidence = confidence_head(res=_res)
      _plddt = _res_confidence[ 'plddt']
      _ptm   = _res_confidence[ 'ptm'  ]
      _loss_plddt = 1.-jnp.mean(_plddt)
      _loss_ptm   = 1.-jnp.mean(_ptm)
      # loss - weighted sum.
      _loss = _loss_colvar*_w_colvar + _loss_plddt*_w_plddt + _loss_ptm*_w_ptm
      # auxiliaries.
      _aux = dict(
        res =_res, 
        loss=dict(colvar=jax.lax.stop_gradient(_loss_colvar), 
                   plddt=jax.lax.stop_gradient(_loss_plddt), 
                   ptm  =jax.lax.stop_gradient(_loss_ptm), 
                  total =jax.lax.stop_gradient(_loss), ), 
        confidence=dict( plddt=jax.lax.stop_gradient(jnp.mean(_plddt)), 
                         ptm  =jax.lax.stop_gradient(jnp.max (_ptm  )), ), 
      )
      return _loss, _aux
    
    # AFEX Optimization prep.
    if feat_afex is None:
      feat_afex = jnp.ones((self.afex.nclus, feat_af.get('seq_length'), self.afex.ntoks))
      self.logger.info(f"COLD start AFEX features, shape {feat_afex.shape}. ")
    else:
      self.logger.info(f"WARM start AFEX features, shape {feat_afex.shape}.")
      
    optimstat = optimizer.init(feat_afex)
    
    _forward_fn = jax.jit(_forward)
    _compile_str = ' (compiling AFEX forward function)'  
    
    # AFEX Optimization loop.
    for _ in range(optim_nsteps):
      self.logger.info(f"Executing AFEX ... step {_:>5}{_compile_str if _==0 else ''}.")

      (loss, aux), grad = jax.value_and_grad(_forward_fn, has_aux=True)(feat_afex)
      optimupdt, optimstat = optimizer.update(updates=grad, state=optimstat)

      # Logging.
      if _ % 1 == 0:
        # Info.
        self.logger.info( " || LOSS: "
                         f"total - { aux['loss']['total' ]:8.4f}; "
                         f"colvar - {aux['loss']['colvar']:8.4f}; "
                         f"plddt - { aux['loss'][ 'plddt']:6.4f}; "
                         f"ptm - {   aux['loss'][ 'ptm'  ]:6.4f}."
                          " || CONFIDENCE: "
                         f"pLDDT - {aux['confidence'][ 'plddt']:6.4f}; "
                         f"pTM - {  aux['confidence'][ 'ptm'  ]:6.4f}.")
        
        # AFEX checkpoint.
        np.save(afex_dir:=os.path.join(self.afex_dir, f"checkpoint{_}.npy"), np.asarray(feat_afex))
        self.logger.info(f"  AFEX checkpoint {afex_dir}")

        # AFEX PDB.
        with open(pdb_dir:=os.path.join(self.pdbs_dir, f"afex{_}.pdb"), 'w') as f:
          p = _af_prot.from_prediction(features =feat_af, 
                                       result   =_u.cast_jnp_to_np(aux['res']), 
                                       b_factors=None, 
                                       remove_leading_feature_dimension=False, ) # False 4 multimer.
          f.write(_af_prot.to_pdb(p))
          self.logger.info(f"  AFEX PDB: {pdb_dir}")

      feat_afex = optax.apply_updates(params=feat_afex, updates=optimupdt)