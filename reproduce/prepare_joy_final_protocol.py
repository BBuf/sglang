"""Freeze final-head plans before their first measurement; never select rows later."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
head = json.loads((ROOT/'evidence/joy-gelu-final-head.json').read_text())['head_sha']
empty = hashlib.sha256(b'').hexdigest()
plans = []


def plan(name, shape, order, purpose):
    contract_name = 'joy-workloads.json' if shape == 'square' else 'joy-workloads-wide.json'
    contract = ROOT/'inputs'/contract_name
    main = 'sgl-joy-gelu-v1-main-high' if shape == 'square' else 'sgl-joy-gelu-v4-wide-main-high'
    candidate = f'sgl-joy-gelu-v4-{shape}-pr-high'
    omni = 'omni-joy-compile-probe1-omni_compile-high' if shape == 'square' else 'omni-joy-gelu-v4-wide-omni-high'
    labels = {'main': 'native resident eager cuDNN main',
              'candidate': 'native resident eager cuDNN committed image GELU candidate',
              'candidate_precleanup': 'native resident eager cuDNN original v1 full source',
              'omni_compile': 'native default regional torch.compile CUDNN_ATTN'}
    result = dict(id=name, created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  purpose=purpose, order=order, quality='high', warmups=2, repeats=4,
                  qualify_each_block=True, quality_references={key:[main,candidate,omni] for key in set(order)},
                  require_pixel_identical=[], configuration_labels={key:labels[key] for key in set(order)},
                  contract='/task/inputs/'+contract_name, frozen_input_sha256=hashlib.sha256(contract.read_bytes()).hexdigest(),
                  candidate_head_sha=head, candidate_diff_sha256=empty,
                  omni_short_quality_reference='/task/outputs/qualification/'+main+'/00-edit-short/output.png',
                  byte_identical_quality_reference={key:(main if key=='main' else candidate) for key in set(order) if key in ('main','candidate')},
                  scope=f'{shape}, same frozen input/checkpoint, 30steps CFG4 BF16, GPU0 batch1/concurrency1; short primary, long/Chinese quality only',
                  primary_performance_evidence=True)
    plans.append(result)
    return result


aa = plan('joy-gelu-final-wide-aa1', 'wide', ['main','main'],
          'Prospective main A/A for the new wide performance cell, before any paired timing of that cell.')
aa['repeats'] = 6
aa['aa_acceptance'] = dict(max_block_median_relative_difference=.005, max_within_block_cv_percent=.5)

cleanup = plan('joy-gelu-final-cleanup1', 'square',
               ['candidate_precleanup','candidate','candidate','candidate_precleanup'],
               'Same-GPU temporally balanced precleanup-v1 vs final-head square model retest. Final eligibility also covers the original8192shape.')
cleanup['candidate_sources'] = {'candidate_precleanup':dict(
    candidate_head_sha='af21e2b989e482cfdce39689126562f2209d53ac',
    candidate_diff_sha256='cc1b771b4d08c2d29e8766bd2b418f6877b125b19ec8a4f0155cb855f10f5da6')}
cleanup['byte_identical_quality_reference']['candidate_precleanup'] = 'sgl-joy-gelu-v1-pr-high'
cleanup['pair_indices'] = {'precleanup_over_final':[[0,1],[3,2]]}
cleanup['cleanup_acceptance'] = {'max_after_over_before_each_pair':1.001,
    'note':'Prospective 0.1% screening ceiling; disclose actual ratios and measured noise. Passing is not automatic publication qualification.'}

for shape in ('square','wide'):
    paired = plan(f'joy-gelu-final-{shape}-paired1', shape,
                  ['main','candidate','omni_compile','omni_compile','candidate','main'],
                  'Final committed source, three native configurations, mirrored temporal order, two fresh processes and eight primary requests per arm. No profiler.')
    paired['pair_indices'] = {'main_over_candidate':[[0,1],[5,4]], 'omni_over_candidate':[[2,1],[3,4]]}
    paired['noise_evidence'] = ('joy-gelu-v1-aa1-analysis.json' if shape=='square' else 'joy-gelu-final-wide-aa1')
    paired['main_endpoint_noise_check'] = {'indices':[0,5], 'max_relative_difference':.005}

for result in plans:
    target = ROOT/'evidence/benchmarks'/(result['id']+'-plan.json')
    assert not target.exists(), target
    target.write_text(json.dumps(result,indent=2)+'\n')
protocol = dict(id='joy-gelu-final-primary1', created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                candidate_head_sha=head, required_quality_review='joy-gelu-v4-complete-quality-review.json',
                square_aa='joy-gelu-v1-aa1-analysis.json',
                plans=[p['id']+'-plan.json' for p in plans],
                exclusions='Tall is quality-only; profiler/Omni denoise diagnostics run separately. No new PR counted by completing this protocol.')
target = ROOT/'evidence/benchmarks/joy-gelu-final-primary1-protocol.json'
assert not target.exists()
target.write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps(protocol,indent=2))
