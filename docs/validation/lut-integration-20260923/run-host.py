import datetime,json,os,pathlib,subprocess,sys,time
root=pathlib.Path('/home/zybi/projects/ifairy.cpp')
evidence=root/'docs/validation/lut-integration-20260923'
scratch=pathlib.Path('/tmp/ifairy-lut-integration')
evidence.mkdir(exist_ok=True)
(evidence/'.gitignore').write_text('!*.log\n!build-*.json\n')
model='/home/zybi/projects/Fairy-plus-minus-i-700M/ifairy.gguf'
records=[]
def run(tag,args,env=None):
    started=datetime.datetime.now(datetime.timezone.utc).isoformat(); t=time.monotonic()
    with (evidence/(tag+'.stdout.log')).open('w') as out,(evidence/(tag+'.stderr.log')).open('w') as err:
        p=subprocess.run(args,cwd=root,env={**os.environ,**(env or {})},stdout=out,stderr=err)
    row=dict(tag=tag,command=args,environment=env or {},started=started,seconds=round(time.monotonic()-t,3),returncode=p.returncode)
    records.append(row); (evidence/(sys.argv[1]+'-commands.json')).write_text(json.dumps(records,indent=2)+'\n')
    print(tag,p.returncode,row['seconds'],flush=True)
    if p.returncode: raise SystemExit(p.returncode)
phase=sys.argv[1]
if phase=='build':
    for name,lut in [('build-rel','OFF'),('build-lut','ON')]:
        run('configure-'+name,['cmake','-B',name,'-DCMAKE_BUILD_TYPE=Release','-DCMAKE_EXPORT_COMPILE_COMMANDS=ON','-DGGML_LEGACY_IFAIRY_CPU_LUT='+lut])
        run('build-'+name,['cmake','--build',name,'-j','8'])
        run('ctest-'+name,['ctest','--test-dir',name,'--output-on-failure','-V'])
    run('build-reference',['cmake','--build','build-lut-reference','-j','8'])
    run('ctest-reference',['ctest','--test-dir','build-lut-reference','--output-on-failure'])
elif phase=='probes':
    cases=[('off','build-lut','batched','0','auto','0'),('auto','build-lut','batched','1','auto','0'),('repeat','build-lut','batched','1','auto','0'),('lut16','build-lut','batched','1','lut16','0'),('lut-c','build-lut','batched','1','lut_c','0'),('serial','build-lut','serial','1','auto','0'),('same-quant-direct','build-lut-reference','batched','0','auto','1')]
    for name,build,mode,lut,impl,act in cases:
        tag='host-probe-'+name
        run(tag,[build+'/bin/test-ifairy-real-model',model,str(scratch/tag),mode,'8'],{'GGML_IFAIRY_LUT':lut,'GGML_IFAIRY_LUT_IMPL':impl,'GGML_IFAIRY_VEC_DOT_ACT_TENSOR':act,'GGML_IFAIRY_LUT_DEBUG':'0'})
    cli=['build-lut/bin/llama-cli','-m',model,'-ngl','0','-t','8','-tb','8','-c','2048','-b','512','-ub','512','--seed','42','--temp','0','-no-cnv','--no-display-prompt','--simple-io']
    run('host-cli-debug',cli+['-p','The capital of France is','-n','1'],{'GGML_IFAIRY_LUT':'1','GGML_IFAIRY_LUT_DEBUG':'1'})
    run('host-cli-story',cli+['-p','Once upon a time, in a small village','-n','128'],{'GGML_IFAIRY_LUT':'1'})
    run('host-cli-long',cli+['-f','docs/validation/local-x86-20260923/long-prompt.txt','-n','256'],{'GGML_IFAIRY_LUT':'1'})
elif phase=='bench':
    for threads,lut in [(4,'0'),(4,'1'),(8,'1'),(8,'0')]:
        run(f'host-bench-{threads}t-'+('on' if lut=='1' else 'off'),['build-lut/bin/llama-bench','-m',model,'-ngl','0','-t',str(threads),'-b','512','-ub','512','-p','128,512','-n','128','-r','3','-o','json'],{'GGML_IFAIRY_LUT':lut,'GGML_IFAIRY_LUT_IMPL':'auto','GGML_IFAIRY_LUT_DEBUG':'0'})
