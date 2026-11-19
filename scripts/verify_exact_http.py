"""REAL TCP smoke for exact-only diagnostic on a FRESH synthetic-only model.

This proves HTTP/process/provenance/cohort contracts, NOT real MovieLens quality or FAISS.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'evidence'/'r20_exact_real_tcp.json')
    args = parser.parse_args()
    sys.path.insert(0,str(ROOT))
    from tests.test_exact_diagnostic import make_synthetic_fixture
    from src.movielens_data import load
    with tempfile.TemporaryDirectory(prefix='r20_exact_tcp_') as dirname:
        tmp=Path(dirname)
        model, ratings, data = make_synthetic_fixture(tmp)
        with socket.socket() as sock:
            sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
        command=[sys.executable,'-m','src.exact_diagnostic_service',
                 '--model',str(model),'--data',str(ratings),'--port',str(port)]
        log=tmp/'server.log'
        env=dict(os.environ);env['PYTHONPATH']=str(ROOT)
        with log.open('w') as stream:
            process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
            try:
                opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
                def get(path):
                    try:
                        with opener.open(f'http://127.0.0.1:{port}'+path,timeout=2) as r:
                            return r.status,json.loads(r.read())
                    except urllib.error.HTTPError as e:
                        return e.code,json.loads(e.read())
                deadline=time.monotonic()+20
                while True:
                    if process.poll() is not None:raise RuntimeError('server exited: '+log.read_text()[-1500:])
                    try:health=get('/health');break
                    except (ConnectionError, OSError):
                        if time.monotonic()>deadline:raise TimeoutError('service startup timed out')
                        time.sleep(.05)
                user=int(data.train[data.train[:,2]>=4,0].min())
                warm=get(f'/recommend/{user}?budget=12')
                cold=get('/recommend/8?budget=12')
                invalid=get('/recommend/999999?budget=12')
                no_positive=get('/recommend/9?budget=12')
                invalid_budget=get('/recommend/1?budget=0')
                seen_warm=set(map(int,data.prefix('test')[data.prefix('test')[:,0]==user,1]))
                seen_cold=set(map(int,data.prefix('test')[data.prefix('test')[:,0]==8,1]))
                assert health[0]==warm[0]==cold[0]==200
                assert invalid[0]==no_positive[0]==invalid_budget[0]==422
                assert health[1]['backend']=='numpy_exact_no_faiss'
                assert warm[1]['method']=='exact_ranker'
                assert cold[1]['method']=='history_exact'
                assert not {row['item'] for row in warm[1]['items']}&seen_warm
                assert not {row['item'] for row in cold[1]['items']}&seen_cold
                result={'status':'PASS_REAL_TCP_SYNTHETIC_FIXTURE',
                        'model_origin':'INDEPENDENTLY_TRAINED_SYNTHETIC_ONLY',
                        'historical_v2_model_live_http':'NOT_RUN_NO_ORIGINAL_RATINGS',
                        'faiss_hnsw_native':'NOT_RUN_FAISS_UNAVAILABLE',
                        'health':health[1],
                        'warm':{'user':user,'method':warm[1]['method'],'item_count':len(warm[1]['items']),'known_seen_returned':0},
                        'history_only':{'user':8,'method':cold[1]['method'],'item_count':len(cold[1]['items']),'known_seen_returned':0},
                        'invalid_user_http_code':invalid[0],
                        'no_positive_history_http_code':no_positive[0],
                        'invalid_budget_http_code':invalid_budget[0]}
            finally:
                process.terminate()
                try:process.wait(timeout=4)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    print(result['status'], 'warm',result['warm']['item_count'],'cold',result['history_only']['item_count'])


if __name__=='__main__':main()
