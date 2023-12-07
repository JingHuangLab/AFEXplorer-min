# Make prediction with AFEX.
# Zilin Song, 20230820
# 

import os

import numpy as np

from typing import Tuple

import jax, jax.numpy as jnp, optax

from absl import app, flags

from alphafold.model import config

from alphafold.common import protein

from afexplore_runner import get_afe_runner, AFExploreRunModel

# DIR: raw_features as input.
flags.DEFINE_string('rawfeat_dir', None, 'Path to directory that stores the raw features.')

# DIR: output.
flags.DEFINE_string('output_dir', None, 'Path to a directory that stores all outputs.')

# DIR: data.
flags.DEFINE_string('afparam_dir', None, 'Path to directory of supporting data / model parameters.')

# Config: number of optimization steps.
flags.DEFINE_integer('nsteps', 10, 'Number of optimization steps.')
# Config-AF: number of MSA clusters
flags.DEFINE_integer('nclust', 512, 'Number of MSA clusters used for featurization, this number scales linearly with memory usage.')

# PRESET: models: monomer only.
flags.DEFINE_enum('model_preset', 'monomer', ['monomer', ],   # 'monomer_casp14', 'monomer_ptm', 'multimer'
                  'Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model')

FLAGS = flags.FLAGS

def afe_fitting(afe_runner: AFExploreRunModel,
                af_features: dict, n_steps: int, 
                ) -> optax.Params:
  """Fit the AFExplore model."""
  afe_weights = jnp.ones(( af_features['msa_feat'].shape[0], 
                           af_features['msa_feat'].shape[1], 
                           af_features['msa_feat'].shape[2], 
                           23, ), )
  optimizer = optax.adam(learning_rate=0.1)
  opt_state = optimizer.init(params=afe_weights)

  print('AFEX started.')
  
  # ------------------------------------------------------------------------------------------------
  def afe_loss_fn(afe_weights: optax.Params, 
                  af_features: dict, 
                  ) -> Tuple[jnp.ndarray, dict]:
    res = afe_runner.predict(af_features, afe_weights, 0)

    # Loss from pLDDT
    # c.f. ./alphafold/common/confidence.py
    plddt_logits = res['predicted_lddt']['logits']
    plddt_bin_width = 1./plddt_logits.shape[-1]
    plddt_bin_centers = jnp.arange(start=.5*plddt_bin_width, stop=1., step=plddt_bin_width, )
    plddt_ca = jnp.sum(jax.nn.softmax(plddt_logits, axis=-1)*plddt_bin_centers[None, :], axis=-1)
    plddt_loss = 1.-jnp.mean(plddt_ca)
    
    # Loss from CV.
    head_ca = res['structure_module']['final_atom_positions'][ 0, 1, :]
    tail_ca = res['structure_module']['final_atom_positions'][-1, 1, :]
    colvar_loss = jnp.square(jnp.linalg.norm(head_ca-tail_ca) - 100.)
    
    return plddt_loss+colvar_loss, (res, jax.lax.stop_gradient(plddt_loss), jax.lax.stop_gradient(colvar_loss))
  # ------------------------------------------------------------------------------------------------

  for _ in range(n_steps): # One optimization step.
    print(f'Runing step {_}', flush=True)
    (loss, aux), grads = jax.value_and_grad(afe_loss_fn, has_aux=True)(afe_weights, af_features)
    updates, opt_state = optimizer.update(updates=grads,  state=opt_state)
    afe_weights = optax.apply_updates(params=afe_weights, updates=updates)

    if _ % 1 == 0:
      print(f'Done Step {_}: plddt_loss: {aux[1]}; cv_loss: {aux[2]}', flush=True)

      p = protein.from_prediction(features=af_features, 
                                  result=aux[0], 
                                  b_factors=None, 
                                  remove_leading_feature_dimension=True, )  # True for Monomer.
      
      with open(os.path.join(FLAGS.output_dir, f'afe_model_{_}.pdb'), 'w') as f:
        f.write(protein.to_pdb(p))

def main(argv):
  """AF2 inference."""
  # Sanity checks: Args count.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  model_runner = get_afe_runner(afparam_dir=FLAGS.afparam_dir, 
                                model_name=config.MODEL_PRESETS[FLAGS.model_preset][0], 
                                num_cluster=FLAGS.nclust, )
  
  # Load featurized MSAs.
  raw_feat = np.load(os.path.join(FLAGS.rawfeat_dir, 'features.pkl'), allow_pickle=True)

  # Process to real features.
  feat = model_runner.process_features(raw_features=raw_feat, 
                                       random_seed=123, )

  # feat = jnp.load(os.path.join(FLAGS.rawfeat_dir, 'processed_features.pkl'), allow_pickle=True)

  afe_fitting(afe_runner=model_runner,
              af_features=feat, 
              n_steps=FLAGS.nsteps, )
  
if __name__ == '__main__':
  flags.mark_flags_as_required([
      'rawfeat_dir',
      'output_dir',
      'afparam_dir',
      'nsteps', 
      'nclust'
  ])

  app.run(main)
