"""Execute the frozen final protocol serially, stopping on quality/noise/cleanup failure."""
import json
import math
import statistics
from pathlib import Path

import run_joy_blocks as harness

ROOT = harness.ROOT
protocol = json.loads((ROOT/'joy-gelu-final-primary1-protocol.json').read_text())
quality = json.loads((ROOT/protocol['required_quality_review']).read_text())
assert quality['all_pass'] and quality['visual_review_pass'] and quality['output_count']==48
assert quality['patch_sha256']=='d471fa9e3522b44bf2c567c5e60ceb264818204437775f32712df31573b77aa8'
aa_square = json.loads((ROOT/protocol['square_aa']).read_text())
assert aa_square['aa']['passed'] and aa_square['source_and_ownership_audit_pass']
status_path = ROOT/'joy-gelu-final-primary1-status.json'
assert not status_path.exists()
status = dict(status='running', protocol=protocol, completed=[], screens=[])


def save():
    status_path.write_text(json.dumps(status,indent=2)+'\n')


try:
    for filename in protocol['plans']:
        plan = json.loads((ROOT/filename).read_text())
        assert plan['candidate_head_sha']==protocol['candidate_head_sha']
        status['active_plan'] = plan['id']
        summaries = []
        for index, config in enumerate(plan['order']):
            status['active_block'] = index
            save()
            harness.run_block(plan,index,config)
            folder = ROOT/'outputs/benchmarks'/f'{plan["id"]}-{index:02d}-{config}'
            summaries.append(json.loads((folder/'summary.json').read_text()))
        medians = [s['p50_seconds'] for s in summaries]
        screen = dict(plan=plan['id'], medians=medians, primary_audit_pending=True)
        if 'aa_acceptance' in plan:
            relative = max(medians)/min(medians)-1
            cv = max(s['cv_percent'] for s in summaries)
            screen.update(aa_relative=relative, aa_max_cv_percent=cv)
            status['screens'].append(screen)
            save()
            assert relative<=plan['aa_acceptance']['max_block_median_relative_difference']
            assert cv<=plan['aa_acceptance']['max_within_block_cv_percent']
        else:
            screen['pairs'] = {label:dict(ratios=[medians[i]/medians[j] for i,j in pairs],
                geometric_mean=math.exp(statistics.mean(math.log(medians[i]/medians[j]) for i,j in pairs)))
                for label,pairs in plan.get('pair_indices',{}).items()}
            if 'main_endpoint_noise_check' in plan:
                i,j=plan['main_endpoint_noise_check']['indices']
                screen['main_endpoint_relative_difference']=max(medians[i],medians[j])/min(medians[i],medians[j])-1
            status['screens'].append(screen)
            save()
            if 'cleanup_acceptance' in plan:
                assert all(1/ratio<=plan['cleanup_acceptance']['max_after_over_before_each_pair']
                           for ratio in screen['pairs']['precleanup_over_final']['ratios']), 'Cleanup regression/noise needs investigation'
        status['completed'].append(plan['id'])
        save()
    status.update(status='raw_primary_complete_audit_and_final_diagnostics_pending',active_plan=None,active_block=None)
except Exception as error:
    status.update(status='failed',error=repr(error))
    raise
finally:
    save()
