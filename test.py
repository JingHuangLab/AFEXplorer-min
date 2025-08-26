import jax.numpy as jnp
import afex.model  as _m
import afex.runner as _r


model_name = 'model_1'

runner = _r.AFEXRunner(work_dir='./run1', 
                       feat_dir='./run1/features.pkl', 
                       afex_config=_m.AFEXConfig(model_name=model_name), 
                       afex_params=_m.AFEXParams(model_name=model_name), )

def colvar(pos: jnp.ndarray) -> jnp.ndarray:
  r"""The CV loss function that should return a scalar tensor as the CV loss.
    
    Args:
      pos (jnp.ndarray): 
        The AF predicted coordinates shape [Nres, 37 ,3], 
        `_res['structure_module']['final_atom_positions']`.
  """
  return 0.



runner.execute(colvar_fn=colvar, optim_nsteps=10)