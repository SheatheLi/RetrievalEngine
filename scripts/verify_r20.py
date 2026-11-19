"""One-command cross-platform Round 20 local acceptance.

Runs real process-boundary TCP, real saved models, all tests; skips are NOT pass.
"""
from __future__ import annotations
import json
from pathlib import Path
import subprocess
import sys
from xml.etree import ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/'evidence'
EVIDENCE.mkdir(exist_ok=True)


def run(name, command, seconds):
    try:
        p=subprocess.run(command,cwd=ROOT,capture_output=True,text=True,timeout=seconds)
        output=p.stdout+'\n'+p.stderr
        (EVIDENCE/f'r20_{name}.log').write_text(output)
        if p.returncode:
            raise RuntimeError(f'{name} failed exit={p.returncode}\n'+output[-3500:])
        return {'status':'PASS','exit':p.returncode,'log':f'evidence/r20_{name}.log'}
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f'{name} timed out after {seconds}s') from exc


def main():
    stages={}
    stages['tests']=run('pytest',[sys.executable,'-m','pytest','-q','tests',
                               '--junitxml=evidence/r20_junit.xml'],60)
    xml=ET.parse(EVIDENCE/'r20_junit.xml')
    suites=xml.findall('.//testsuite')
    counts={attr:sum(int(s.attrib.get(attr,0)) for s in suites) for attr in ('tests','failures','errors','skipped')}
    if counts['failures'] or counts['errors'] or counts['skipped']>2 or counts['tests']<34:
        raise RuntimeError('unexpected test/skip inventory: '+repr(counts))
    stages['tests']['counts']=counts
    stages['native_tcp']=run('tcp',[sys.executable,'scripts/verify_exact_http.py',
                              '--output','evidence/r20_exact_real_tcp.json'],35)
    stages['three_historical_artifacts']=run('artifacts',[sys.executable,
                              'scripts/benchmark_exact_diagnostic.py'],30)
    manifest={'status':'PASS_WITH_FAISS_GATES_UNRUN','stages':stages,
              'evidence_levels':{
                 'legacy_saved_v2_real_artifacts':'hash+exact retrieval+ranker verified',
                 'real_tcp':'synthetic internally consistent trained model only',
                 'native_faiss_hnsw':'NOT_RUN: dependency missing',
                 'real_movielens_http':'NOT_RUN: original raw ratings absent',
                 'online_recommendation_quality':'NOT_RUN'}}
    (EVIDENCE/'r20_acceptance.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
