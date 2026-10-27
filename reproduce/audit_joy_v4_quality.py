"""Audit the complete declared model-quality run from archived immutable inputs/PNGs."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'evidence'
QUAL = EVIDENCE/'qualification'
p = argparse.ArgumentParser()
p.add_argument('status',type=Path)
p.add_argument('--output',type=Path,required=True)
a = p.parse_args()


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


status = read(a.status)
assert status['status']=='numeric_quality_passed_visual_review_pending'
assert status['completed']==['main','pr','omni']
plan=status['plan']
outputs=[]
for stage in plan['order']:
    source=stage['source']
    arm='omni' if source=='omni' else 'sgl'
    metadata=read(EVIDENCE/'block-runs'/f'joy-gelu-v4-{source}'/'metadata.json')
    expected_head=plan['omni_sha'] if source=='omni' else plan['candidate_head_sha'] if source=='pr' else plan['baseline_sha']
    expected_diff=plan['candidate_diff_sha256'] if source=='pr' else hashlib.sha256(b'').hexdigest()
    assert metadata['source_sha']==expected_head and metadata['diff_sha256']==expected_diff
    assert digest(EVIDENCE/'block-runs'/f'joy-gelu-v4-{source}'/'source.patch')==expected_diff
    assert metadata['status']=='quality_outputs_generated_not_yet_compared' and not metadata['benchmark']
    for shape in stage['shapes']:
        contract_record=plan['contracts'][shape]
        local_contract=ROOT/Path(contract_record['path']).relative_to('/task')
        assert digest(local_contract)==contract_record['sha256']
        contract=read(local_contract)
        assert digest(ROOT/contract['reference_image']['path'])==contract['reference_image']['sha256']
        for quality in stage['qualities']:
            name=f'{arm}-joy-gelu-v4-{shape}-{source}-{quality}'
            folder=QUAL/name
            assert digest(folder/'contract.json')==contract_record['sha256']
            rows=read(folder/'results.json')
            assert len(rows)==4
            prompts=contract['prompts']+[contract['prompts'][0]]
            for index,(row,prompt) in enumerate(zip(rows,prompts)):
                request_dir=folder/f'{index:02d}-{prompt["id"]}'
                assert Path(row['request']).name==request_dir.name
                request=read(request_dir/'request.json')
                assert request['image']==contract['reference_image']
                form=request['form']
                for key in ['seed','num_inference_steps','guidance_scale','negative_prompt','generator_device']:
                    assert form[key]==str(contract[key])
                assert form['model']==contract['model_path'] and form['n']=='1'
                assert form['prompt']==prompt['prompt'] and form['size']==f'{contract["width"]}x{contract["height"]}'
                assert not row['benchmark'] and not row['profiled']
                if arm=='sgl':
                    assert form['quality']==quality
                image=request_dir/'output.png'
                assert digest(image)==row['sha256']
                outputs.append(dict(source=source,shape=shape,quality=quality,prompt=request_dir.name,
                                    path=str(image.relative_to(ROOT)),sha256=row['sha256']))
            if arm=='sgl':
                assert rows[0]['sha256']==rows[3]['sha256']

checks=[]
for check in status['checks']:
    gate=read(QUAL/(check['candidate']+'-quality.json'))
    assert gate['all_pass'] and len(gate['results'])==4*len(check['references'])
    assert {r['reference'] for r in gate['results']}==set(check['references'])
    assert len({(r['prompt'],r['reference']) for r in gate['results']})==len(gate['results'])
    for row in gate['results']:
        assert digest(QUAL/check['candidate']/row['prompt']/'output.png')==row['candidate_sha256']
        assert digest(QUAL/row['reference']/row['prompt']/'output.png')==row['reference_sha256']
        assert row['passes_cross_framework_gate']
        if check['pixel_identical_required']:
            assert row['pixel_identical']
    for reference in check['references']:
        rows=[r for r in gate['results'] if r['reference']==reference]
        checks.append(dict(candidate=check['candidate'],reference=reference,
                           pixel_identical=all(r['pixel_identical'] for r in rows),
                           min_ssim=min(r['ssim'] for r in rows),
                           min_psnr_db=min((r['psnr_db'] for r in rows if r['psnr_db'] is not None),default=None),
                           worst_ssim_prompt=min(rows,key=lambda r:r['ssim'])['prompt']))
assert len(outputs)==48 and len(status['checks'])==10
square_before=read(QUAL/'sgl-joy-gelu-v1-pr-high/results.json')
square_after=read(QUAL/'sgl-joy-gelu-v4-square-pr-high/results.json')
result=dict(numeric_audit_pass=True,visual_review_pass=False,output_count=len(outputs),
            patch_sha256=plan['candidate_diff_sha256'],plan=plan,outputs=outputs,comparisons=checks,
            square_precleanup_v1_vs_v4_pngs_identical=[r['sha256'] for r in square_before]==[r['sha256'] for r in square_after],
            scope='Quality and source/input/actual PNG audit only, no performance claim; inspect actual images before allowing primary timing.')
a.output.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('plan','outputs')},indent=2))
