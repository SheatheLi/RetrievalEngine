"""Start a real inference process and verify model-bound TCP responses."""
from pathlib import Path
import argparse,json,socket,subprocess,sys,time,urllib.request,urllib.error


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,required=True)
    parser.add_argument('--data',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    # Select a registered user from the training prefix, without using test labels.
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from src.movielens_data import load
    data=load(args.data)
    user=int(data.train[data.train[:,2]>=4,0].min())
    with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
    command=[sys.executable,'-m','src.recommender_service','--model',str(args.model),
             '--data',str(args.data),'--port',str(port)]
    log=args.output.with_suffix('.server.log')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w') as stream:
        process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT)
        try:
            opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
            def get(path):
                with opener.open(f'http://127.0.0.1:{port}'+path,timeout=3) as r:return json.loads(r.read())
            ready=None
            until=time.monotonic()+20
            while time.monotonic()<until:
                if process.poll() is not None:raise RuntimeError('inference server exited; read '+str(log))
                try:ready=get('/health');break
                except OSError:time.sleep(.05)
            if ready is None:raise TimeoutError('inference server not ready')
            response=get(f'/recommend/{user}?budget=100')
            ids=[x['item'] for x in response['items']]
            prefix=data.prefix('test');seen=set(prefix[prefix[:,0]==user,1].tolist())
            assert len(ids)==10 and len(set(ids))==10 and set(ids).isdisjoint(seen)
            assert ready['input_sha256']==data.input_sha256
            errors=[]
            for path in ['/recommend/999999','/recommend/1?budget=9']:
                try:get(path)
                except urllib.error.HTTPError as e:errors.append(e.code)
            assert errors==[422,422]
            args.output.write_text(json.dumps({'command':command,'health':ready,'response':response,
                'known_seen_recommended':0,'cold_and_invalid_statuses':errors,'result':'PASS_REAL_LOOPBACK_HTTP'},indent=2)+'\n')
            print('PASS_REAL_LOOPBACK_HTTP user',user)
        finally:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=3)


if __name__=='__main__':main()
