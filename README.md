# AlphaFoldTools: versatile, flexible, and interoperable protein structure predictions with AlphaFold
## Installation
If you have a working AF2 environment, install the following packages depends on your JAX/JAXLIB 
version in the environment. For more info, see the corresponding GitHub repos and find a compatible 
release.
```bash
chex
optax
```
If install from scratch, the following provides a non-Docker approach, depends on your cuda version:
```bash
conda create --name af2
conda activate af2
conda install -y -c conda-forge openmm==7.5.1 cudatoolkit=11.2 pdbfixer
pip install -r requirements.txt --no-cache-dir
pip install --upgrade --no-cache-dir jax==0.3.25 jaxlib==0.3.25+cuda11.cudnn805 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install --no-cache-dir chex==0.1.5 optax==0.1.3
```
If encountered a error related to "ptxas version string" arises, a working solution is to:
```bash
conda install -c nvidia cuda-nvcc
```
This will:
1. Bump your Numpy version to 1.25. Some type hints discrepancies in the AF source code are now 
raised as errors by Numpy, e.g., `np.int` -> `np.int32`, `np.object` -> `object`, which is easy to 
fix: just change the AF source.  
2. Disable SciPy 1.7.0. due to incompatiblity to Numpy 1.25. This will disable pLDDT score calculations
by the out-of-box AF2 inference pipeline. It only impacts the per-Residue pLDDT calculations after the
AF2 inference. AFExplore inference does not rely on SciPy for per-Residue pLDDT calculation.
## Running AFExplore. 
**AFExplore is compatible solely to AF-Monomer.**
Note that the AF2 parameter directories are renamed to `--afparam_dir`.
- ./afexplore/af_feat.py: Running featurization from fasta files. Almost the same way you would do 
  for conventional AF2 runs. Note that some binary paths are given explicitly and the option FLAGS
  have been re-organized.
- ./afexplore/afexplore_optim.py: This is the gradients dynamics loop. 
- ./afexplore/afexplore_runner.py: The surrogate model around `alphafold.model.model.RunModel` to exec
  AFExplore gradients dynamics.
## License
AFExplore inherits the license from AF2: no additional term was added.
## NOTE - Z.S.: 
**[Zilin Song](https://github.com/ZL-Song)** (song.zilin@outlook.com)
- No AF2 source codes were modified: reproducing AFExplore given the reference presented here with your own AF
  should be simple. 
- OOM error is normal for longer sequences, e.g., a 444 AA monomer chain once requested ~47 GiB memory. 
  Such situation is beyond my expertise: AFExplore requires that the entire AF2 graph (which is huge) to be 
  loaded ready for reverse mode gradient computation. 
- As a DL amateur, I can hardly keep track of all DL methods in the field that may share similarities
  to AFExplore. The following two works are the references that I found to be most relavent to what is 
  presented here.

  1. [AF-Profile](https://github.com/patrickbryant1/AFProfile): Very interesting work on improving AF-Multimer
    predictions by adding a bias to the input and train against a pLDDT loss. This work provides conceputual 
    inspirations. 

  2. [GNNExplainer](https://arxiv.org/abs/1903.03894): Also applies weights to the input features 
    but for a different purpose (XAI). This work provides technical inspirations. 
