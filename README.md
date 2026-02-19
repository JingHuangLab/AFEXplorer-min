# AFEXplorer-min: The minimum implementation of AFEXplorer for conditioned prediction of protein structures.
**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)  
## Installation
If you have a working AF2 environment, install the following packages depends on your JAX/JAXLIB 
version in the environment. For more info, see the corresponding GitHub repos and find a compatible 
release.
```bash
chex
optax
```
If installing from scratch, the following provides a non-Docker approach, depends on your cuda version:
```bash
conda create --name af2
conda activate af2
conda install -y -c conda-forge openmm==7.5.1 cudatoolkit=11.2 pdbfixer
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --no-cache-dir chex==0.1.5 optax==0.1.3
```
If an error related to "ptxas version string" arises, a working solution is to:
```bash
conda install -c nvidia cuda-nvcc
```
This will:
1. Bump your NumPy version to 1.25. Some type hints discrepancies in the AF source code are now 
raised as errors by Numpy, e.g., `np.int` -> `np.int32`, `np.object` -> `object`, which is easy to 
fix: just change the AF source.  
2. Disable SciPy 1.7.0. due to incompatiblity to Numpy 1.25. This will disable pLDDT score calculations
by the out-of-box AF2 inference pipeline. It only impacts the per-Residue pLDDT calculations after the
AF2 inference. AFExplorer inference does not rely on SciPy for the pLDDT calculations.
## NOTE - Z.S.
- This implementation is compatible to both **AF** and **AF-Multimer**.
- I have only made the following changes to the AF source code:  
`alphafold/model/modules_multimer.py` lines 427-429, 520-522, 647-649.
- In principle, the AFEX code can also be migrated to ColabFold (which essentially enables more efficient MSA featurizations and shares the identical inference pipeline with AF/AF-Multimer). However, we did not test AFEX with ColabFold at this moment.  
## References
If referring to the AF-Monomer implementation of AFEX, you are welcomed to cite: 
```latex
@article{song2024afex,
  title={Conditioned protein structure prediction},
  author={Xie, Tengyu and Song, Zilin and Huang, Jing},
  journal={PRX Life},
  volume={2},
  number={4},
  pages={043001},
  year={2024},
  publisher={APS}
}
```
