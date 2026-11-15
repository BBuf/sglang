"""Frozen native API inputs; match actual H.264 CRF18/ultrafast output encoding."""
def build_payload(arm, contract, item, directory):
    prompt = contract['prompts'][item['prompt_id']]
    payload = dict(prompt=prompt,seed=item['seed'],num_inference_steps=item['steps'],
                   guidance_scale=item['guidance_scale'],flow_shift=item['flow_shift'],
                   width=item['width'],height=item['height'],num_frames=item['frames'],
                   fps=item['fps'],negative_prompt='',num_outputs_per_prompt=1)
    if arm=='sgl':
        payload.update(model='/task/cache/nomas-native-common02',quality=item.get('quality','lossless'),
                       generator_device='cpu',max_sequence_length=1280,
                       segments=[dict(frames=item['frames'],caption=prompt)],prompt_format='json',
                       enhance_prompt=False,output_compression=64,x264_preset='ultrafast',
                       output_path=str(directory/'native-output'),perf_dump_path=str(directory/'perf.json'))
    else:
        assert arm=='omni'
        # The frozen Omni native model rejects video_codec_options in extra_args.
        # Its native API encoder already uses CRF18, ultrafast, threads=0.
    return payload
