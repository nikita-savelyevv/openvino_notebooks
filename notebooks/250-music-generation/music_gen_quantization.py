from collections import namedtuple
import gc
from pathlib import Path
from typing import Optional, Tuple
import warnings

from IPython.display import Audio
from openvino import Core, convert_model, PartialShape, save_model, Type
import numpy as np
import soundfile as sf
import torch
from torch.jit import TracerWarning
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

# Ignore tracing warnings
warnings.filterwarnings("ignore", category=TracerWarning)

text_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

SAMPLING_RATE = 32000
SAMPLE_LENGTH = 8
N_TOKENS = {8: 403}[SAMPLE_LENGTH]

core = Core()
DEVICE = "AUTO"

models_dir = Path("./models")
t5_ir_path = models_dir / "t5.xml"
musicgen_0_ir_path = models_dir / "mg_0.xml"
musicgen_ir_path = models_dir / "mg.xml"
audio_decoder_onnx_path = models_dir / "encodec.onnx"
audio_decoder_ir_path = models_dir / "encodec.xml"


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder_ir, config):
        super().__init__()
        self.encoder = core.compile_model(encoder_ir, DEVICE)
        self.config = config

    def forward(self, input_ids, **kwargs):
        last_hidden_state = self.encoder(input_ids)[self.encoder.outputs[0]]
        last_hidden_state = torch.tensor(last_hidden_state)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state)


class MusicGenWrapper(torch.nn.Module):
    def __init__(self, music_gen_lm_0_ir, music_gen_lm_ir, config, num_codebooks, build_delay_pattern_mask,
                 apply_delay_pattern_mask):
        super().__init__()
        self.music_gen_lm_0 = core.compile_model(music_gen_lm_0_ir, DEVICE)
        self.music_gen_lm = core.compile_model(music_gen_lm_ir, DEVICE)
        self.config = config
        self.num_codebooks = num_codebooks
        self.build_delay_pattern_mask = build_delay_pattern_mask
        self.apply_delay_pattern_mask = apply_delay_pattern_mask

    def forward(
            self,
            input_ids: torch.LongTensor = None,
            encoder_hidden_states: torch.FloatTensor = None,
            encoder_attention_mask: torch.LongTensor = None,
            past_key_values: Optional[Tuple[torch.FloatTensor]] = None,
            **kwargs
    ):
        if past_key_values is None:
            model = self.music_gen_lm_0
            arguments = (input_ids, encoder_hidden_states, encoder_attention_mask)
        else:
            model = self.music_gen_lm
            arguments = (input_ids, encoder_hidden_states, encoder_attention_mask, *past_key_values)

        output = model(arguments)
        return CausalLMOutputWithCrossAttentions(
            logits=torch.tensor(output[model.outputs[0]]),
            past_key_values=tuple([output[model.outputs[i]] for i in range(1, 97)]),
        )


class AudioDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder_ir, config):
        super().__init__()
        self.decoder = core.compile_model(decoder_ir, DEVICE)
        self.config = config
        self.output_type = namedtuple("AudioDecoderOutput", ["audio_values"])

    def decode(self, output_ids, audio_scales):
        output = self.decoder(output_ids)[self.decoder.outputs[0]]
        return self.output_type(audio_values=torch.tensor(output))


def infer_pytorch_model(model, input_text, device="cpu"):
    assert SAMPLE_LENGTH * model.config.audio_encoder.frame_rate + 3 == N_TOKENS
    assert model.config.audio_encoder.sampling_rate == SAMPLING_RATE

    inputs = text_processor(
        text=[input_text],
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=N_TOKENS)

    sf.write("original.wav", audio_values[0, 0].cpu().numpy(), samplerate=SAMPLING_RATE)


def convert_to_ir(model):
    inputs = text_processor(
        text=["80s pop track with bassy drums and synth"],
        return_tensors="pt",
    )

    #
    # 1. Convert Text Encoder
    #
    if not t5_ir_path.exists():
        t5_ov = convert_model(model.text_encoder, example_input={'input_ids': inputs['input_ids']})
        save_model(t5_ov, t5_ir_path)
        del t5_ov
        gc.collect()

    #
    # 2. Convert MusicGen Language Model
    #
    # Set model config `torchscript` to True, so the model returns a tuple as output
    model.decoder.config.torchscript = True

    if not musicgen_0_ir_path.exists() or not musicgen_ir_path.exists():
        decoder_input = {
            'input_ids': torch.ones(8, 1, dtype=torch.int64),
            'encoder_hidden_states': torch.ones(2, 12, 1024, dtype=torch.float32),
            'encoder_attention_mask': torch.ones(2, 12, dtype=torch.int64),
        }
        mg_ov_0_step = convert_model(model.decoder, example_input=decoder_input)

        save_model(mg_ov_0_step, musicgen_0_ir_path)
        del mg_ov_0_step
        gc.collect()

        # Add `past_key_values` to the converted model signature
        decoder_input['past_key_values'] = tuple(
            [(
                torch.ones(2, 16, 1, 64, dtype=torch.float32),
                torch.ones(2, 16, 1, 64, dtype=torch.float32),
                torch.ones(2, 16, 12, 64, dtype=torch.float32),
                torch.ones(2, 16, 12, 64, dtype=torch.float32),
            )] * 24
        )
        mg_ov = convert_model(model.decoder, example_input=decoder_input)

        for input in mg_ov.inputs[3:]:
            input.get_node().set_partial_shape(PartialShape([-1, 16, -1, 64]))
            input.get_node().set_element_type(Type.f32)
        mg_ov.validate_nodes_and_infer_types()

        save_model(mg_ov, musicgen_ir_path)
        del mg_ov
        gc.collect()

    #
    # 3. Convert Audio Decoder
    #
    if not audio_decoder_onnx_path.exists():
        class AudioDecoder(torch.nn.Module):
            def __init__(self, model):
                super().__init__()
                self.model = model

            def forward(self, output_ids):
                return self.model.decode(output_ids, [None])

        audio_decoder_input = {'output_ids': torch.ones(1, 1, 4, N_TOKENS - 3, dtype=torch.int64), }

        with torch.no_grad():
            torch.onnx.export(
                model=AudioDecoder(model.audio_encoder),
                args=audio_decoder_input,
                f=audio_decoder_onnx_path,
                input_names=['output_ids', ],
                output_names=['decoded_audio'],
                dynamic_axes={
                    'output_ids': {3: 'sequence_length'},
                    'decoded_audio': {2: 'audio_values'}
                }
            )

    # Now we can convert the model to OpenVINO IR
    if not audio_decoder_ir_path.exists():
        audio_decoder_ov = convert_model(str(audio_decoder_onnx_path))

        save_model(audio_decoder_ov, audio_decoder_ir_path)
        del audio_decoder_ov
        gc.collect()


def infer_ov_model(model, input_text):
    # Embedding the converted models into the original pipeline
    text_encode_ov = TextEncoderWrapper(t5_ir_path, model.text_encoder.config)
    musicgen_decoder_ov = MusicGenWrapper(
        musicgen_0_ir_path,
        musicgen_ir_path,
        model.decoder.config,
        model.decoder.num_codebooks,
        model.decoder.build_delay_pattern_mask,
        model.decoder.apply_delay_pattern_mask
    )
    audio_encoder_ov = AudioDecoderWrapper(audio_decoder_ir_path, model.audio_encoder.config)

    del model.text_encoder
    del model.decoder
    del model.audio_encoder
    gc.collect()

    model.text_encoder = text_encode_ov
    model.decoder = musicgen_decoder_ov
    model.audio_encoder = audio_encoder_ov

    inputs = text_processor(
        text=[input_text],
        return_tensors="pt",
    )

    audio_values = model.generate(**inputs.to("cpu"), do_sample=True, guidance_scale=3, max_new_tokens=N_TOKENS)

    sf.write("openvino.wav", audio_values[0, 0].cpu().numpy(), samplerate=SAMPLING_RATE)


model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small", torchscript=True, return_dict=False).to("cpu").eval()
# infer_pytorch_model(model, "80s pop track with bassy drums and synth")
# convert_to_ir(model)
infer_ov_model(model, "80s pop track with bassy drums and synth")

