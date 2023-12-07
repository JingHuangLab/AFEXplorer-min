#!/bin/bash
#SBATCH -N 1
#SBATCH -c 1
#SBATCH -p yuhuan
#SBATCH --mem=30GB
#SBATCH --exclusive

# conda activate af2

seq=./data/ABL1/ABL1.fasta
output_dir=./data/ABL1
database_dir=/u/xiety/database/alphafold-2.2.0/
date=1700-01-01

python ./afexplore/af_feat.py \
       --fasta_paths "$seq"        \
       --output_dir $output_dir    \
       --uniref90_database_path    $database_dir/uniref90/uniref90.fasta \
       --mgnify_database_path      $database_dir/mgnify/mgy_clusters.fa \
       --bfd_database_path         $database_dir/bfd/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt \
       --uniref30_database_path    $database_dir/uniclust30/uniclust30_2018_08/uniclust30_2018_08 \
       --pdb70_database_path       $database_dir/pdb70/pdb70 \
       --obsolete_pdbs_path        $database_dir/pdb_mmcif/obsolete.dat \
       --template_mmcif_dir        $database_dir/pdb_mmcif/mmcif_files \
       --max_template_date $date \
       --model_preset monomer