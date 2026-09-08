"""Test exact reverse GDN traversal against materialized flip/shift reference."""
import json
import os
import torch
import triton
from sglang.multimodal_gen.runtime.models.dits import sana_wm_components as wm


def backward(qr, kr, v, beta, decay, q=None, k=None):
    b,h,d,n=qr.shape
    t=beta.shape[2];s=n//t
    # flip/shift's cat -> from_time reshape materializes row-major K/V.
    kr=kr.contiguous().view(b,h,d,t,s)
    v=v.contiguous().view(b,h,d,t,s)
    qr=qr.view(b,h,d,t,s)
    main=q is not None
    if main:
        q=q.view(b,h,d,t,s)
        k=k.contiguous().view(b,h,d,t,s)
    beta=beta.unsqueeze(3) if beta.ndim==4 else beta.view(b,h,t,1,1)
    decay=decay.view(b,h,t,1,1)
    state=torch.zeros(b,h,d,d,device=qr.device,dtype=qr.dtype)
    z=torch.zeros(b,h,d,1,device=qr.device,dtype=qr.dtype) if main else None
    nums=[None]*t;dens=[None]*t
    for i in range(t-1,-1,-1):
        if i+1<t:
            j=i+1;kt=kr[:,:,:,j];vt=v[:,:,:,j]
            bt=beta[:,:,j];gt=decay[:,:,j]
            state=state*gt
            if main:z=z*gt
            dv=(vt-torch.matmul(state,kt))*bt
            state=state+torch.matmul(dv,kt.transpose(-1,-2))
            if main:
                key=k[:,:,:,j]
                dz=(1.0-torch.matmul(z.transpose(-1,-2),key))*bt
                z=z+torch.matmul(key,dz.transpose(-1,-2))
        # The first reverse update has all-zero K/V/beta and unit decay.
        # Its state stays zero; keep Q matmuls to preserve nonfinite-Q behavior.
        nums[i]=torch.matmul(state,qr[:,:,:,i])
        if main:dens[i]=torch.matmul(z.transpose(-1,-2),q[:,:,:,i])
    num=torch.stack(nums,dim=2).permute(0,1,3,2,4).reshape(b,h,d,n)
    if not main:return num
    den=torch.stack(dens,dim=2).permute(0,1,3,2,4).reshape(b,h,1,n)
    return num,den


def main_cached(q,k,v,qr,kr,beta,decay,init_state_kv=None,init_state_z=None):
    (nf,df),state=wm._gdn_scan_forward_stateful(q,k,v,qr,kr,beta,decay,
        init_state_kv=init_state_kv,init_state_z=init_state_z,
        return_components=True,return_state=True)
    nb,db=backward(qr,kr,v,beta,decay,q,k)
    return (nf+nb)/(df+db+1e-6),state


def cam_cached(qr,kr,v,beta,decay,init_state_kv=None):
    f,state=wm._single_path_delta_scan_forward_stateful(qr,kr,v,beta,decay,
        init_state_kv=init_state_kv,return_state=True)
    return f+backward(qr,kr,v,beta,decay),state


def exact(a,b):
    if isinstance(a,tuple):return all(exact(x,y) for x,y in zip(a,b,strict=True))
    return torch.equal(a,b)


def run():
    rows=[]
    with torch.inference_mode():
        for t in [1,3,4]:
            for layout in ['dn','nd']:
                for seed in range(2):
                    torch.manual_seed(seed)
                    b,h,d,s=1,20,112,880;n=t*s
                    def rand(scale=.03):
                        z=torch.randn((b,h,d,n) if layout=='dn' else (b,h,n,d),device='cuda')*scale
                        return z if layout=='dn' else z.transpose(-1,-2)
                    q,k,v,qr,kr=[rand() for _ in range(5)]
                    beta=torch.rand(b,h,t,s,device='cuda')*.02
                    decay=torch.rand(b,h,t,device='cuda')*.5+.4
                    state=torch.randn(b,h,d,d,device='cuda')*.001
                    z=torch.randn(b,h,d,1,device='cuda')*.001
                    for kind in ['main','cam']:
                        if kind=='main':
                            def ref():return wm._gdn_scan_cached(q,k,v,qr,kr,beta,decay,init_state_kv=state,init_state_z=z)
                            def opt():return main_cached(q,k,v,qr,kr,beta,decay,state,z)
                        else:
                            def ref():return wm._single_path_delta_scan_cached(qr,kr,v,beta,decay,init_state_kv=state)
                            def opt():return cam_cached(qr,kr,v,beta,decay,state)
                        a=ref();c=opt()
                        row={'t':t,'layout':layout,'seed':seed,'kind':kind,'exact':exact(a,c),'max_abs':(a[0]-c[0]).abs().max().item()}
                        if seed==0 and t in [3,4]:
                            row['reference_ms']=float(triton.testing.do_bench_cudagraph(ref,rep=100))
                            row['candidate_ms']=float(triton.testing.do_bench_cudagraph(opt,rep=100))
                        rows.append(row);print(json.dumps(row),flush=True)
    with open(os.environ['SANA_REVERSE_RESULT'],'w') as f:json.dump(rows,f,indent=2)

if __name__=='__main__':run()
