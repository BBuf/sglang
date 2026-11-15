"""Admission-only native tensor/step receipts; never used for timing visits."""
import hashlib
import json
import os
import time
from pathlib import Path

folder = os.environ.get('NOMAS_AUDIT_DIR')
arm = os.environ.get('NOMAS_AUDIT_ARM')
if folder and arm in ('sgl', 'omni'):
    import torch

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    receipt = folder / f'pid-{os.getpid()}.jsonl'
    counter = 0
    current = None

    def record(event, **values):
        row = dict(event=event, arm=arm, pid=os.getpid(), wall_ns=time.time_ns(), request=current, **values)
        with receipt.open('a') as f:
            f.write(json.dumps(row) + '\n')

    def tensor_info(value):
        value = value.detach().contiguous().cpu()
        return dict(shape=list(value.shape), dtype=str(value.dtype),
                    sha256=hashlib.sha256(value.view(torch.uint8).numpy().tobytes()).hexdigest())

    def save_inputs(latents, embeddings, mask, **values):
        name = f'pid-{os.getpid()}-request-{current}-inputs.pt'
        tensors = dict(initial_noise=latents.detach().cpu(),text_embeddings=embeddings.detach().cpu(),text_mask=mask.detach().cpu())
        torch.save(tensors,folder/name)
        record('native_inputs',tensors={k:tensor_info(v) for k,v in tensors.items()},tensor_file=name,**values)

    try:
        if arm == 'sgl':
            from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.nomas.stages import NomasDenoisingStage
            from sglang.multimodal_gen.runtime.models.dits.nomas import NomasTransformer3DModel

            stage_forward = NomasDenoisingStage.forward
            predict = NomasTransformer3DModel.predict

            def observed_forward(self,batch,server_args):
                global current,counter
                counter += 1;current=counter
                assert batch.session is None, 'Observer admits offline requests only'
                conditions=batch.extra['nomas_conditions'];segments=batch.extra['nomas_segments']
                assert len(conditions)==len(segments)==1
                condition=conditions[0]
                record('request_begin',seed=batch.seed,steps=batch.num_inference_steps,quality=batch.quality,
                       prompt_sha256=hashlib.sha256((batch.prompt[0] if isinstance(batch.prompt,list) else batch.prompt).encode()).hexdigest())
                save_inputs(batch.latents,condition.text_states,condition.text_mask,
                            timesteps=batch.scheduler.timesteps.cpu().tolist(),sigmas=batch.scheduler.sigmas.cpu().tolist(),
                            actions=[condition.action_start,condition.text_offsets,condition.video_windows],
                            segments=[[s.start,s.end] for s in segments])
                result=stage_forward(self,batch,server_args)
                record('denoise_complete',latent=tensor_info(result.latents))
                return result

            def observed_predict(self,latent,timestep,*args,**kwargs):
                record('native_predict',timestep=timestep.detach().cpu().tolist(),commit=bool(kwargs.get('collect_cache',False)))
                return predict(self,latent,timestep,*args,**kwargs)

            NomasDenoisingStage.forward=observed_forward
            NomasTransformer3DModel.predict=observed_predict
        else:
            from vllm_omni.diffusion.models.hy_nomas.pipeline_hy_nomas import HyNomasPipeline

            prepare=HyNomasPipeline.prepare_next_chunk
            forward_chunk=HyNomasPipeline._forward_chunk
            scheduler_step=HyNomasPipeline.step_scheduler

            def observed_prepare(self,state):
                global current,counter
                result=prepare(self,state)
                counter+=1;current=counter
                request=state.extra['hy_nomas_request']
                record('request_begin',seed=request.seed,steps=request.steps,
                       prompt_sha256=hashlib.sha256(request.prompts[0].encode()).hexdigest())
                save_inputs(state.latents,state.prompt_embeds,state.extra['hy_nomas_text_mask'],
                            timesteps=state.scheduler.timesteps.cpu().tolist(),sigmas=state.scheduler.sigmas.cpu().tolist(),
                            offsets=state.extra['hy_nomas_intra_offsets'],windows=state.extra['hy_nomas_intra_windows'])
                return result

            def observed_chunk(self,state,latents,timestep,*args,**kwargs):
                record('native_predict',timestep=timestep.detach().cpu().tolist(),commit=bool(kwargs.get('commit',False)))
                return forward_chunk(self,state,latents,timestep,*args,**kwargs)

            def observed_step(self,state,noise_pred,**kwargs):
                result=scheduler_step(self,state,noise_pred,**kwargs)
                if state.step_in_chunk==state.extra['hy_nomas_request'].steps:
                    record('denoise_complete',latent=tensor_info(state.latents))
                return result

            HyNomasPipeline.prepare_next_chunk=observed_prepare
            HyNomasPipeline._forward_chunk=observed_chunk
            HyNomasPipeline.step_scheduler=observed_step
        record('observer_installed',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    except Exception as exc:
        record('observer_install_failed',error=repr(exc))
        raise
