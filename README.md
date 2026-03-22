# AFEXplorer-min: The minimum implementation of AFEXplorer for conditional prediction of protein structures.
**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)  
## Installation
If you have a working AF2 environment, install the following packages depends on your JAX/JAXLIB 
version in the environment. For more info, see the corresponding GitHub repos and find a compatible 
release.
```bash
chex
optax
```
If install from scratch, the following provides a non-Docker approach, depends on your cuda version (below are for CUDA 11.2):
```bash
conda create --name af2
conda activate af2
conda install -y -c conda-forge openmm==7.5.1 cudatoolkit=11.2 pdbfixer
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 optax==0.1.7 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```
If an error related to "ptxas version string" arises, try `module load` your cuda library.
## NOTE - Z.S.
- This implementation is compatible to both **AF** and **AF-Multimer**.
- I have only made the following changes to the AF source code:  
`alphafold/model/modules_multimer.py` lines 427-429, 520-522, 647-649.  
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