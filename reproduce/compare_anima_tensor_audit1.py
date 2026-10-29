"""Compare actual unmodified native Anima request inputs and observations."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
root=Path('/task')
base=root/'outputs/diagnostics/anima-tensor-audit1'
out=root/'evidence/preflight/anima-runtime-input-audit1.json'
assert not out.exists()
status=json.loads((root/'anima-tensor-audit1-status.json').read_text())
assert status['status']=='raw_complete' and all(x['byte_exact'] for x in status['comparisons'])

def read(arm,kind):
    result=[]
    for p in sorted((base/arm).glob('*/'+kind+'-*.pt')):
        d=torch.load(p,weights_only=True)
        d['file']=str(p);d['sha256']=hashlib.sha256(p.read_bytes()).hexdigest();result.append(d)
    return result

def metric(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    delta=a.float()-b.float()
    return dict(exact=torch.equal(a,b),shape=list(a.shape),dtype_a=str(a.dtype),dtype_b=str(b.dtype),
        max_abs=float(delta.abs().max()),relative_l2=float(delta.norm()/a.float().norm().clamp_min(1e-30)))

sgc=read('sgl','sgl-conditioner')[-8:]
omc=read('omni','omni-conditioner')[-8:]
sgs=read('sgl','scheduler-step')[-8:]
oms=read('omni','scheduler-step')[-8:]
sgt=read('sgl','sgl-text-encoder')
omt=read('omni','omni-text-encoder')[-8:]
assert len(sgc)==len(omc)==len(sgs)==len(oms)==len(omt)==8
rows=[]
for i,prompt in enumerate(['short','long','chinese','short-repeat']):
    conditions=[]
    for j,branch in enumerate(['positive','negative']):
        a,b=sgc[2*i+j],omc[2*i+j]
        q=omt[2*i+j]
        matches=[x for x in sgt if torch.equal(x['inputs']['input_ids'],q['inputs']['input_ids'])
                 and torch.equal(x['inputs']['attention_mask'],q['inputs']['attention_mask'])]
        assert len(matches)==1,(prompt,branch,len(matches))
        p=matches[0]
        conditions.append(dict(branch=branch,qwen_ids=metric(p['inputs']['input_ids'],q['inputs']['input_ids']),
            qwen_mask=metric(p['inputs']['attention_mask'],q['inputs']['attention_mask']),
            qwen_hidden=metric(p['output'],q['output']),
            conditioner_inputs={k:metric(a['inputs'][k],b['inputs'][k]) for k in a['inputs']},
            conditioning_output=metric(a['output'],b['output']),
            source_files=[a['file'],b['file'],p['file'],q['file']]))
    a,b=sgs[2*i],oms[2*i]
    assert a['step_index_before'] in [None,0] and b['step_index_before'] in [None,0]
    rows.append(dict(prompt=prompt,conditions=conditions,
        initial_noise=metric(a['sample'],b['sample']),timesteps=metric(a['timesteps'],b['timesteps']),
        sigmas=metric(a['sigmas'],b['sigmas']),first_noise_prediction=metric(a['model_output'],b['model_output']),
        first_scheduler_output=metric(a['result'],b['result']),
        final_scheduler_output=metric(sgs[2*i+1]['result'],oms[2*i+1]['result'])))
result=dict(scope='Actual native runtime observation; all8 output PNGs byte-exact unobserved originals',
    requests=rows,observed_outputs_byte_exact=True,
    noise_and_schedule_exact=all(all(x[k]['exact'] for k in ['initial_noise','timesteps','sigmas']) for x in rows),
    token_ids_masks_exact=all(all(c[k]['exact'] for k in ['qwen_ids','qwen_mask']) and
        all(c['conditioner_inputs'][k]['exact'] for k in ['target_input_ids','target_attention_mask','source_attention_mask'])
        for x in rows for c in x['conditions']),
    note='SGL native text-embedding cache reuses negative and repeated short encoding; match Qwen inputs by exact IDs/mask, retain all saved forwards. Warmup records excluded explicitly (SGL2 and Omni1 conditioner/step records). Numerical hidden-state differences remain measured, not input override.',
    script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='requests'}))
for row in rows:
    print(row['prompt'], 'Qwen/conditioner relativeL2',[(c['branch'],c['qwen_hidden']['relative_l2'],c['conditioning_output']['relative_l2']) for c in row['conditions']], 'firstnoise',row['first_noise_prediction']['relative_l2'])
