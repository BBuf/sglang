"""Publish verified three-round metadata while preserving incomplete transport status."""
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path

root = Path(__file__).resolve().parents[1]
secret = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{25,}|github_pat_[A-Za-z0-9_]{30,}|hf_[A-Za-z0-9]{25,}|https?://[^\s/:]+:[^\s/@]+@)')
configs = [
    ('nomas', 'nomas-followup41', 'nomas-mean-37999', 'f50be8b83e165d7fc639acad22c969ec52607f16', 'nomas-PR50602-body41-unpublished.md', 'EVIDENCE41_COMMIT'),
    ('qwen21', 'qwen21-followup47', 'qwen21-prefix-15b', 'f04c6e867dfd5229fbb4665ba33049db64807d75', 'qwen21-PR50519-body47-unpublished.md', 'EVIDENCE47_COMMIT'),
]
for model, archive, tree, parent, draft, marker in configs:
    out = root / 'evidence/public' / tree
    assert subprocess.check_output(['git', '-C', str(out), 'rev-parse', 'HEAD'], text=True).strip() == parent
    assert not subprocess.check_output(['git', '-C', str(out), 'status', '--porcelain'])
    receipt = json.loads((root / 'evidence' / (archive + '-raw.receipt.json')).read_text())
    partial = json.loads((root / 'evidence/preflight' / (archive + '-partial-member-review53.json')).read_text())
    required = {'evidence/benchmarks/' + model + '-three-rounds35.json', 'scripts/analyze_protocol_rounds35.py'}
    selected = []
    for row in receipt['files']:
        rel = row['path']
        if not rel.startswith(('inputs/', 'scripts/', 'evidence/')):
            continue
        p = root / rel
        if p.suffix not in ('.py', '.json', '.jsonl', '.txt', '.log', '.yaml') or not p.is_file():
            continue
        assert p.stat().st_size == row['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest() == row['sha256'], rel
        selected.append(row)
    assert required <= {x['path'] for x in selected}
    if model == 'nomas':
        assert 'evidence/quality/nomas-long-scope29-summary36.json' in {x['path'] for x in selected}
    for rel in ('evidence/' + archive + '-raw.receipt.json',
                'evidence/preflight/' + archive + '-partial-member-review53.json',
                'evidence/results-matrix-before-publication48.json', 'scripts/prepare_metadata53.py'):
        p = root / rel
        selected.append(dict(path=rel, bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    if model == 'nomas':
        p = root / 'evidence/preflight/nomas-source-exclusions46.json'
        selected.append(dict(path=str(p.relative_to(root)), bytes=p.stat().st_size, sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
    changes = []
    for row in selected:
        p = root / row['path']
        raw = p.read_bytes()
        text = raw.decode('utf-8')
        assert not secret.search(text), row['path']
        target = out / row['path']
        if target.exists():
            assert target.read_bytes() == raw, row['path']
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
        changes.append(row['path'])
    notice = '''**Evidence availability update:** the three-round report below includes every measured latency sample and every round, with original source heads unchanged. Complete raw archives were created on the 6000 host with per-file SHA256 receipts. SSH interrupted the local archive copies on 2026-11-15; this supplement publishes the verified reports, commands and hash manifests available locally. The remaining raw-file mirror is pending reconnection. Previously published actual main/PR images and videos remain available. The supplemental raw-file mirror remains incomplete; the previous raw archive and generated comparisons remain available.\n\n'''
    body = (root / 'evidence' / draft).read_text()
    body = body.split('<!-- pr-states:start -->')[0].rstrip() + '\n'
    body = notice + body
    (root / 'evidence' / (model + '-body53-unpublished.md')).write_text(body)
    body = body.replace('https://github.com/BBuf/sglang/blob/' + marker + '/', '')
    assert marker not in body
    (out / 'README.md').write_text(body)
    changes.append('README.md')
    manifest = dict(model=model, parent=parent, raw_archive_bytes=receipt['bytes'], raw_archive_sha256=receipt['sha256'],
                    local_archive_complete=False, partial_bytes=partial['partial_bytes'],
                    individually_verified_complete_members=partial['complete_members_verified'],
                    files=selected, note='All report cells and original round samples are retained. This commit does not claim a complete local raw archive or restored PR qualification.')
    (out / 'metadata53-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    changes.append('metadata53-manifest.json')
    subprocess.run(['git', '-C', str(out), 'add', '--', *changes], check=True)
    subprocess.run(['git', '-C', str(out), 'diff', '--cached', '--check'], check=True)
    elements = [dict(path=rel, mode='100644', type='blob', content=(out / rel).read_text()) for rel in changes]
    payload = dict(repository_full_name='BBuf/sglang',
                   base_tree_sha=subprocess.check_output(['git', '-C', str(out), 'rev-parse', parent + '^{tree}'], text=True).strip(),
                   tree_elements=elements)
    payload_path = root / 'evidence/preflight' / (model + '-metadata53-tree-payload.json')
    payload_path.write_text(json.dumps(payload, ensure_ascii=False))
    tree_sha = subprocess.check_output(['git', '-C', str(out), 'write-tree'], text=True).strip()
    result = dict(model=model, parent=parent, tree=tree_sha, files=len(elements), payload_bytes=payload_path.stat().st_size,
                  branch=subprocess.check_output(['git', '-C', str(out), 'branch', '--show-current'], text=True).strip())
    (root / 'evidence/preflight' / (model + '-metadata53-staging.json')).write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
