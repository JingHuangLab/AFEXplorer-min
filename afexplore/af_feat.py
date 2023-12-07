# AF2 data pipeline for preparing features for model input. 
# Zilin Song, 20230810
# 

"""Prepare features for AF2 inputs, customized from AF2 interface."""

import os
import pickle
import pathlib
from absl import app, flags, logging

from alphafold.data import pipeline, templates
from alphafold.data.tools import hhsearch

logging.set_verbosity(logging.INFO)

# PATHs: input (fasta).
flags.DEFINE_list('fasta_paths', None,  'Paths to FASTA files, each containing a prediction target '
                                        'that will be folded one after another. If a FASTA file '
                                        'contains multiple sequences, then it will be folded as a '
                                        'multimer. Paths should be separated by commas. All FASTA '
                                        'paths must have a unique basename as the basename is used '
                                        'to name the output directories for each prediction.', )

# DIR: output.
flags.DEFINE_string('output_dir', None, 'Path to a directory that will store the results.')

 
# PATH: execs.
flags.DEFINE_string('jackhmmer_binary_path', '/apps/anaconda/anaconda3/envs/alphafold/bin/jackhmmer', 'Path to the JackHMMER executable.')
flags.DEFINE_string(  'hhblits_binary_path', '/apps/anaconda/anaconda3/envs/alphafold/bin/hhblits',   'Path to the   HHblits executable.')
flags.DEFINE_string( 'hhsearch_binary_path', '/apps/anaconda/anaconda3/envs/alphafold/bin/hhsearch',  'Path to the  HHsearch executable.')  # model_preset==monomer only
flags.DEFINE_string(   'kalign_binary_path', '/apps/anaconda/anaconda3/envs/alphafold/bin/kalign',    'Path to the    Kalign executable.')

# PATH: databases.
flags.DEFINE_string(  'uniref90_database_path', None, 'Path to the Uniref90 database for use by JackHMMER.')
flags.DEFINE_string(    'mgnify_database_path', None, 'Path to the   MGnify database for use by JackHMMER.')
flags.DEFINE_string(       'bfd_database_path', None, 'Path to the      BFD database for use by   HHblits.') 
flags.DEFINE_string(  'uniref30_database_path', None, 'Path to the UniRef30 database for use by   HHblits.')
flags.DEFINE_string(     'pdb70_database_path', None, 'Path to the    PDB70 database for use by  HHsearch.') # model_preset==monomer only
# PATH: obsolete PDB fixes.
flags.DEFINE_string('obsolete_pdbs_path', None, 'Path to file containing a mapping from obsolete PDB IDs to the PDB IDs of their replacements.')


# Below are immutable keywords.
# PATH & PROP: tempelates.
flags.DEFINE_string('template_mmcif_dir', None, 'Path to a directory with template mmCIF structures, each named <pdb_id>.cif')
flags.DEFINE_string('max_template_date',  None, 'Maximum template release date to consider. Important if folding historical test sets.')

# PRESET: Dataset:
flags.DEFINE_enum('db_preset', 'full_dbs', ['full_dbs', ],  # 'reduced_dbs'
                  'Choose preset MSA database configuration - smaller genetic database config (reduced_dbs) or full genetic database config  (full_dbs)')
# PRESET: models.
flags.DEFINE_enum('model_preset', 'monomer', ['monomer', ], # 'monomer_casp14', 'monomer_ptm', 'multimer'
                  'Choose preset model configuration - the monomer model, the monomer model with extra ensembling, monomer model with pTM head, or multimer model')

FLAGS = flags.FLAGS
MAX_TEMPLATE_HITS = 0

def main(argv):
  """Main protocols."""
  # Sanity checks: Args count.
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')
  
  # Sanity checks: required tools.
  for tool_name in ('jackhmmer', 'hhblits', 'hhsearch', 'kalign'):

    if not FLAGS[f'{tool_name}_binary_path'].value:
      raise ValueError(f'Could not find path to the "{tool_name}" binary.'
                        'Make sure it is installed on your system.')
  
  # Sanity checks: non-duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]

  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  # Config model runtime.
  template_searcher = hhsearch.HHSearch(binary_path=FLAGS.hhsearch_binary_path,
                                        databases=[FLAGS.pdb70_database_path], )
  
  template_featurizer = templates.HhsearchHitFeaturizer(mmcif_dir=FLAGS.template_mmcif_dir,
                                                        max_template_date=FLAGS.max_template_date,
                                                        max_hits=MAX_TEMPLATE_HITS,
                                                        kalign_binary_path=FLAGS.kalign_binary_path,
                                                        release_dates_path=None,
                                                        obsolete_pdbs_path=FLAGS.obsolete_pdbs_path, )

  data_pipeline = pipeline.DataPipeline(jackhmmer_binary_path=FLAGS.jackhmmer_binary_path,
                                        hhblits_binary_path=FLAGS.hhblits_binary_path,
                                        uniref90_database_path=FLAGS.uniref90_database_path,
                                        mgnify_database_path=FLAGS.mgnify_database_path,
                                        bfd_database_path=FLAGS.bfd_database_path,
                                        uniref30_database_path=FLAGS.uniref30_database_path,
                                        small_bfd_database_path=None, # for use_small_bfd == False
                                        template_searcher=template_searcher,
                                        template_featurizer=template_featurizer,
                                        use_small_bfd=False,
                                        use_precomputed_msas=False, )
  
  # Only one fasta in AF2-monomer.
  fasta_path = FLAGS.fasta_paths[0]

  # Make paths and dirs.
  output_msa_dir   = os.path.join(FLAGS.output_dir, 'msas')
  output_feat_path = os.path.join(FLAGS.output_dir, 'raw_features.pkl')

  if not os.path.exists(output_msa_dir):
    os.makedirs(output_msa_dir)

  features_dict = data_pipeline.process(input_fasta_path=fasta_path, 
                                        msa_output_dir=output_msa_dir, )
  
  with open(output_feat_path, 'wb') as f:
    pickle.dump(features_dict, f, protocol=4)

if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'uniref90_database_path',
      'uniref30_database_path',
      'mgnify_database_path',
      'obsolete_pdbs_path',
      'template_mmcif_dir',
      'max_template_date',
  ])

  app.run(main)