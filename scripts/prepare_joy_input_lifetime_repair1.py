"""Freeze an isolated repair experiment without changing any failed head evidence."""
import ast
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
repo = root/'candidates/joy-input-lifetime-repair1'
head = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
assert head == '03368cf1c325e526184b8fb223443e1ae931b13c'
assert not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'])
now = datetime.datetime.now(datetime.timezone.utc).isoformat()
def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    assert not p.exists(),p
    p.write_text(json.dumps(value,indent=2)+'\n')
model = 'python/sglang/multimodal_gen/runtime/models/dits/joy_image.py'
test = 'test/registered/kernels/ops/diffusion/test_model_fast_paths.py'
original = subprocess.check_output(['git','-C',str(repo),'show','4c135ea92d6b4cbb0ead69bb39976f7dbe995455:'+model],text=True)
current = (repo/model).read_text()
class Normalize(ast.NodeTransformer):
    def visit_Name(self,node):
        if node.id == 'fc_in': node.id='linear'
        return node
    def visit_FunctionDef(self,node):
        if node.name == '_joy_image_mlp' and isinstance(node.body[0],ast.Expr) and isinstance(node.body[0].value,ast.Constant):
            node.body.pop(0)
        return self.generic_visit(node)
assert ast.dump(Normalize().visit(ast.parse(current))) == ast.dump(Normalize().visit(ast.parse(original)))
red_folder = root/'evidence/diagnostics/joy-input-lifetime-repair1-red-attempt2'
green_folder = root/'evidence/diagnostics/joy-input-lifetime-repair1-green-attempt1'
for folder,code in [(red_folder,1),(green_folder,0)]:
    receipt=json.loads((folder/'receipt.json').read_text())
    assert receipt['expected_result_observed'] and receipt['returncode']==code and receipt['skipped']==0
    for name,sha in receipt['files_sha256'].items(): assert digest(folder/name)==sha
patch=subprocess.check_output(['git','-C',str(repo),'diff','--full-index','HEAD^','HEAD','--'])
assert patch==(green_folder/'source.patch').read_bytes()
admission=root/'evidence/preflight/joy-input-lifetime-repair1-tests-admission.json'
save(admission,dict(passed=True,at=now,source_sha=head,source_clean=True,tests_passed=1,tests_skipped=0,
    red_before_fix=True,green_after_fix=True,red_receipt_sha256=digest(red_folder/'receipt.json'),
    green_receipt_sha256=digest(green_folder/'receipt.json'),patch_sha256=hashlib.sha256(patch).hexdigest(),
    original_runtime_AST_equal_after_local_name_and_docstring_normalization=True,
    source_files_sha256={p:digest(repo/p) for p in [model,test]},
    performance_admitted=False,not_new_PR_or_optimization_slot=True))
save(root/'evidence/preflight/joy-input-lifetime-repair1-self-review.json',dict(at=now,head=head,
    base='b08c24e9630c63790c9f86b48d6c7fc10a0c4301',changed_files=[model,test],
    complete_diff_reviewed=True,cleanup_complete_before_model_measurement=True,
    implementation='Release temporary image input by replacing the input binding with fused hidden before fc_out; retain output projection wrapper.',
    regression='Actual SM120 temporary input must be dead at output projection entry. Unfixed production failed; repaired production and original numerical/fallback/compile/LoRA checks passed.',
    retained_constraints=['Original eligibility and arithmetic unchanged','No new API/configuration/dependency','No unrelated lint/format edits','No model timing conclusion from this operator regression'],
    source_diff_sha256=hashlib.sha256(patch).hexdigest()))
with (root/'shared-files.jsonl').open('a') as stream:
    for p in [model,test]:
        stream.write(json.dumps(dict(at=now,task='joy-input-lifetime-repair1',branch='codex/joy-input-lifetime-repair-20261030',head=head,path=p,
            scope='Isolated remediation of existing Joy cleanup; no new optimization PR counted'),ensure_ascii=False)+'\n')
# Copy existing audited admission/archive programs only to give this distinct source its own immutable experiment namespace.
for src,dst in [('admit_joy_current_head1.py','admit_joy_input_lifetime_repair1.py'),('archive_joy_current_head1.py','archive_joy_input_lifetime_repair1.py')]:
    data=(root/'scripts'/src).read_text().replace('joy-current-head-', 'joy-input-lifetime-repair1-')
    out=root/'scripts'/dst;assert not out.exists();out.write_text(data)
plans=[]
for shape in ['square','wide']:
    for phase in ['aa','primary']:
        source=root/'evidence/plans'/f'joy-current-head-{shape}-{phase}1.json'
        p=json.loads(source.read_text()); old=p['id'];run=f'joy-input-lifetime-repair1-{shape}-{phase}1'
        p.update(id=run,created_at=now,current_head=head,repair_of_head='b08c24e9630c63790c9f86b48d6c7fc10a0c4301',
            purpose='Fresh isolated minimal lifetime repair vs original tested Joy/main/native Omni. Preserve failed official head experiment; no replacement PR or new optimization slot.',
            prior_failed_square_primary='joy-current-head-square-primary1',
            prior_failed_result_not_superseded_or_discarded=True,
            source_self_review='evidence/preflight/joy-input-lifetime-repair1-self-review.json')
        p['required_admissions']=['evidence/preflight/joy-input-lifetime-repair1-tests-admission.json']
        if phase=='primary': p['required_admissions'].append(f'evidence/preflight/joy-input-lifetime-repair1-{shape}-aa1-admission.json')
        p['sha256'].pop('scripts/admit_joy_current_head1.py')
        for name in ['admit_joy_input_lifetime_repair1.py','archive_joy_input_lifetime_repair1.py']:
            p['sha256']['scripts/'+name]=digest(root/'scripts'/name)
        p['sha256']['evidence/preflight/joy-input-lifetime-repair1-tests-admission.json']=digest(admission)
        if phase=='primary':
            p['configurations']['candidate_current'].update(repo='/task/candidates/joy-input-lifetime-repair1',head=head)
            p['candidate_sources']['candidate_current']['candidate_head_sha']=head
            p['configuration_labels']['candidate_current']='native resident eager SYNC0 lifetime repair 03368cf1'
        # Preserve every frozen quality, timing, sample count and numerical gate.
        for key in ['order','warmups','repeats','quality','quality_references','byte_identical_quality_reference']:
            assert p[key]==json.loads(source.read_text())[key]
        save(root/'evidence/plans'/(run+'.json'),p)
        plans.append(dict(run=run,sha256=digest(root/'evidence/plans'/(run+'.json'))))
save(root/'evidence/preflight/joy-input-lifetime-repair1-preparation.json',dict(at=now,source_sha=head,plans=plans,
    gates_and_sample_counts_unchanged=True,failed_results_retained=True,no_new_performance_data=True,
    tests_admission_sha256=digest(admission)))
print(json.dumps(dict(head=head,plans=plans,tests_admission=str(admission)),indent=2))
