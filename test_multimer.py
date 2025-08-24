import jax.numpy as jnp
import afex.model_multimer  as _m
import afex.runner_multimer as _r


model_name = 'model_1_multimer_v3'

runner = _r.AFEXMultimerRunner(work_dir='./run0', 
                               feat_dir='./run0/features.pkl', 
                               afex_config=_m.AFEXMultimerConfig(model_name=model_name), 
                               afex_params=_m.AFEXMultimerParams(model_name=model_name), )

def colvar(pos: jnp.ndarray) -> jnp.ndarray:
  r"""pos - [Nres, 37 ,3]."""



runner.execute(colvar_fn=lambda x: 0., optim_nsteps=10)


# import alphafold.model.model as _af_m
# runner = _af_m.RunModel(config=_m.AFEXMultimerConfig(model_name=model_name), 
#                         params=_m.AFEXMultimerParams(model_name=model_name), )
# import numpy as np
# res = runner.predict(feat=np.load('./run0/features.pkl', allow_pickle=True), random_seed=0)
# import pickle as pkl
# with open ('./run0/afresult.pkl', 'wb') as f:
#   pkl.dump(obj=res, file=f)
# print(res['plddt'], res['ptm'], res['iptm'])