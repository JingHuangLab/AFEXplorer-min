import jax.numpy as jnp
import afex.model_multimer  as _m
import afex.runner_multimer as _r


model_name = 'model_1_multimer_v3'

runner = _r.AFEXMultimerRunner(work_dir='./run0', 
                               feat_dir='./run0/features.pkl', 
                               afex_config=_m.AFEXMultimerConfig(model_name=model_name), 
                               afex_params=_m.AFEXMultimerParams(model_name=model_name), )

def colvar(pos: jnp.ndarray) -> jnp.ndarray:
  r"""The CV loss function that should return a scalar tensor as the CV loss.
    
    Args:
      pos (jnp.ndarray): 
        The AF-Multimer predicted coordinates shape [Nres, 37 ,3], 
        `_res['structure_module']['final_atom_positions']`.
  """
  return 0.



runner.execute(colvar_fn=colvar, optim_nsteps=10)