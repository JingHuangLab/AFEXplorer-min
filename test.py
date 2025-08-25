import jax.numpy as jnp
import afex.model  as _m
import afex.runner as _r


model_name = 'model_1'

runner = _r.AFEXRunner(work_dir='./run1', 
                       feat_dir='./run1/features.pkl', 
                       afex_config=_m.AFEXConfig(model_name=model_name), 
                       afex_params=_m.AFEXParams(model_name=model_name), )

def colvar(pos: jnp.ndarray) -> jnp.ndarray:
  r"""pos - [Nres, 37 ,3]."""



runner.execute(colvar_fn=lambda x: 0., optim_nsteps=10)