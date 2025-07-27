
import afex.model_multimer  as _m
import afex.runner_multimer as _r


model_name = 'model_1_multimer_v3'

runner = _r.AFEXMultimerRunner(work_dir='./run0', 
                               feat_dir='./run0/features.pkl', 
                               afex_config=_m.AFEXMultimerConfig(model_name=model_name), 
                               afex_params=_m.AFEXMultimerParams(model_name=model_name), )

runner.execute(colvar_fn=lambda x: 0., )