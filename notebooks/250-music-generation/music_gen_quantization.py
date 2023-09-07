import datetime
import pickle
from collections import namedtuple
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Optional, Tuple
import warnings

import onnx

from IPython.display import Audio
from openvino import Core, convert_model, PartialShape, save_model, Type
import numpy as np
import soundfile as sf
import torch
from torch.jit import TracerWarning
from tqdm import tqdm
from transformers import AutoProcessor, MusicgenForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions, CausalLMOutputWithCrossAttentions

import nncf
import datasets

from nncf import QuantizationPreset, ModelType
from nncf.quantization.advanced_parameters import OverflowFix

# Ignore tracing warnings
warnings.filterwarnings("ignore", category=TracerWarning)

text_processor = AutoProcessor.from_pretrained("facebook/musicgen-small")

SAMPLING_RATE = 32000
SAMPLE_LENGTH = 8
N_TOKENS = {2: 103, 4: 203, 8: 403}[SAMPLE_LENGTH]

core = Core()
DEVICE = "AUTO"

models_dir = Path(f"./models/{SAMPLE_LENGTH}")
t5_ir_path = models_dir / "t5.xml"
musicgen_0_ir_path = models_dir / "mg_0.xml"
musicgen_ir_path = models_dir / "mg.xml"
audio_decoder_onnx_path = models_dir / "encodec.onnx"
audio_decoder_ir_path = models_dir / "encodec.xml"

CALIBRATION_CACHE_PATH = "calibration_data/{}.pkl"


COLLECT_CALIBRATION_DATA = False
text_encoder_calibration_data = []
text_decoder_0_calibration_data = []
text_decoder_calibration_data = []
audio_decoder_calibration_data = []


@contextmanager
def calibration_data_collection_context():
    global COLLECT_CALIBRATION_DATA
    COLLECT_CALIBRATION_DATA = True
    yield
    COLLECT_CALIBRATION_DATA = False


text_decoder_inference_times = []


class TextEncoderWrapper(torch.nn.Module):
    def __init__(self, encoder_ir, config):
        super().__init__()
        self.encoder = core.read_model(encoder_ir)
        self.encoder_compiled = core.compile_model(self.encoder, DEVICE)
        self.config = config

    def forward(self, input_ids, **kwargs):
        if COLLECT_CALIBRATION_DATA:
            text_encoder_calibration_data.append(input_ids)
        last_hidden_state = self.encoder_compiled(input_ids)[self.encoder_compiled.outputs[0]]
        last_hidden_state = torch.tensor(last_hidden_state)
        return BaseModelOutputWithPastAndCrossAttentions(last_hidden_state=last_hidden_state)


class MusicGenWrapper(torch.nn.Module):
    def __init__(self, music_gen_lm_0_ir, music_gen_lm_ir, config, num_codebooks, build_delay_pattern_mask,
                 apply_delay_pattern_mask):
        super().__init__()
        self.music_gen_lm_0 = core.read_model(music_gen_lm_0_ir)
        self.music_gen_lm_0_compiled = core.compile_model(self.music_gen_lm_0, DEVICE)
        self.music_gen_lm = core.read_model(music_gen_lm_ir)
        self.music_gen_lm_compiled = core.compile_model(self.music_gen_lm, DEVICE)
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
            model = self.music_gen_lm_0_compiled
            arguments = (input_ids, encoder_hidden_states, encoder_attention_mask)
            if COLLECT_CALIBRATION_DATA:
                text_decoder_0_calibration_data.append(arguments)
        else:
            model = self.music_gen_lm_compiled
            arguments = (input_ids, encoder_hidden_states, encoder_attention_mask, *past_key_values)
            if COLLECT_CALIBRATION_DATA:
                text_decoder_calibration_data.append(arguments)

        start_time = datetime.datetime.now()
        output = model(arguments)
        end_time = datetime.datetime.now()
        if past_key_values is not None:
            text_decoder_inference_times.append((end_time - start_time).total_seconds())
        return CausalLMOutputWithCrossAttentions(
            logits=torch.tensor(output[model.outputs[0]]),
            past_key_values=tuple([output[model.outputs[i]] for i in range(1, 97)]),
        )


class AudioDecoderWrapper(torch.nn.Module):
    def __init__(self, decoder_ir, config):
        super().__init__()
        self.decoder = core.read_model(decoder_ir)
        self.decoder_compiled = core.compile_model(self.decoder, DEVICE)
        self.config = config
        self.output_type = namedtuple("AudioDecoderOutput", ["audio_values"])

    def decode(self, output_ids, audio_scales):
        if COLLECT_CALIBRATION_DATA:
            audio_decoder_calibration_data.append(output_ids)
        output = self.decoder_compiled(output_ids)[self.decoder_compiled.outputs[0]]
        return self.output_type(audio_values=torch.tensor(output))


class AudioDecoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, output_ids):
        return self.model.decode(output_ids, [None])


def infer_model(model, input_text, save_filename, device="cpu"):
    global text_decoder_inference_times
    assert SAMPLE_LENGTH * model.config.audio_encoder.frame_rate + 3 == N_TOKENS
    assert model.config.audio_encoder.sampling_rate == SAMPLING_RATE

    print("Inferencing")

    inputs = text_processor(
        text=[input_text],
        return_tensors="pt",
    )

    start_time = datetime.datetime.now()
    text_decoder_inference_times = []
    audio_values = model.generate(**inputs.to(device), do_sample=True, guidance_scale=3, max_new_tokens=N_TOKENS)
    print(f"Decoder inference time: {sum(text_decoder_inference_times):.2f} "
          f"({len(text_decoder_inference_times)} calls)")
    print(f"Total inference time: {(datetime.datetime.now() - start_time).total_seconds():.2f}")

    sf.write(save_filename, audio_values[0, 0].cpu().numpy(), samplerate=SAMPLING_RATE)


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
        with torch.no_grad():
            torch.onnx.export(
                model=AudioDecoder(model.audio_encoder),
                args={'output_ids': torch.ones(1, 1, 4, N_TOKENS - 3, dtype=torch.int64), },
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


def convert_to_ov_model(model):
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
    return model


def collect_calibration_data(n_samples):
    global text_encoder_calibration_data, text_decoder_0_calibration_data, text_decoder_calibration_data, \
        audio_decoder_calibration_data

    calibration_cache_path = Path(CALIBRATION_CACHE_PATH.format(n_samples))

    calibration_dataset = datasets.load_dataset("google/MusicCaps", split="train", streaming=True).take(
        n_samples)
    if not calibration_cache_path.exists():
        if not calibration_cache_path.parent.exists():
            calibration_cache_path.parent.mkdir(parents=True)
        with calibration_data_collection_context():
            for data_item in tqdm(calibration_dataset, total=n_samples,
                                  desc="Collecting calibration data"):
                inputs = text_processor(text=[data_item["caption"]], return_tensors="pt").to("cpu")
                model.generate(**inputs, do_sample=True, guidance_scale=3, max_new_tokens=N_TOKENS)

        # with open(calibration_cache_path, 'wb') as f:
        #     pickle.dump((text_encoder_calibration_data, text_decoder_0_calibration_data, text_decoder_calibration_data,
        #                  audio_decoder_calibration_data), f)
    else:
        print("Loading calibration data")
        with open(calibration_cache_path, "rb") as f:
            text_encoder_calibration_data, text_decoder_0_calibration_data, text_decoder_calibration_data, \
                audio_decoder_calibration_data = pickle.load(f)


def quantize(model, save_dir, n_calibration_samples, preset, sq_alpha):
    save_dir = Path(save_dir)

    #
    # Collect calibration data
    #

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
    model.text_encoder = text_encode_ov
    model.decoder = musicgen_decoder_ov
    model.audio_encoder = audio_encoder_ov

    #
    # Quantize text encoder
    #
    # compressed_text_encoder_path = save_dir / "text_encoder.xml"
    # if not compressed_text_encoder_path.exists():
    #     compressed_text_encoder = nncf.quantize(
    #         model.text_encoder.encoder,
    #         calibration_dataset=nncf.Dataset(text_encoder_calibration_data),
    #         preset=QuantizationPreset.PERFORMANCE,
    #         subset_size=len(text_encoder_calibration_data),
    #         model_type=ModelType.TRANSFORMER,
    #         ignored_scope=None,
    #         advanced_parameters=nncf.AdvancedQuantizationParameters(
    #             overflow_fix=OverflowFix.DISABLE,
    #             smooth_quant_alpha=0.95
    #         )
    #     )
    #     save_model(compressed_text_encoder, compressed_text_encoder_path)
    # model.text_encoder = TextEncoderWrapper(compressed_text_encoder_path, model.text_encoder.config)

    #
    # Quantize text decoder 0
    #
    # compressed_text_decoder_0_path = save_dir / "text_decoder_0.xml"
    # if not compressed_text_decoder_0_path.exists():
    #     compressed_text_decoder_0 = nncf.quantize(
    #         model.decoder.music_gen_lm_0,
    #         calibration_dataset=nncf.Dataset(text_decoder_0_calibration_data),
    #         preset=QuantizationPreset.MIXED,
    #         subset_size=len(text_decoder_0_calibration_data),
    #         model_type=ModelType.TRANSFORMER,
    #         ignored_scope=None,
    #         advanced_parameters=nncf.AdvancedQuantizationParameters(
    #             overflow_fix=OverflowFix.DISABLE,
    #             smooth_quant_alpha=0.95
    #         )
    #     )
    #     save_model(compressed_text_decoder_0, compressed_text_decoder_0_path)
    # model.decoder.music_gen_lm_0 = core.read_model(compressed_text_decoder_0_path)
    # model.decoder.music_gen_lm_0_compiled = core.compile_model(model.decoder.music_gen_lm_0, DEVICE)

    #
    # Quantize text decoder full
    #
    compressed_text_decoder_path = save_dir / "text_decoder.xml"
    if not compressed_text_decoder_path.exists():
        collect_calibration_data(n_calibration_samples)
        print("Quantizing")
        compressed_text_decoder = nncf.quantize(
            model.decoder.music_gen_lm,
            calibration_dataset=nncf.Dataset(text_decoder_calibration_data),
            preset=preset,
            subset_size=len(text_decoder_calibration_data),
            model_type=ModelType.TRANSFORMER,
            ignored_scope=None,
            advanced_parameters=nncf.AdvancedQuantizationParameters(
                overflow_fix=OverflowFix.DISABLE,
                smooth_quant_alpha=sq_alpha
            )
        )

        # print("Compressing weights")
        # compressed_text_decoder = nncf.compress_weights(model.decoder.music_gen_lm)

        save_model(compressed_text_decoder, compressed_text_decoder_path)
    model.decoder.music_gen_lm = core.read_model(compressed_text_decoder_path)
    model.decoder.music_gen_lm_compiled = core.compile_model(model.decoder.music_gen_lm, DEVICE)

    #
    # Quantize audio decoder
    #
    # compressed_audio_decoder_path = save_dir / "audio_decoder.xml"
    # if not compressed_audio_decoder_path.exists():
    #     compressed_audio_decoder = nncf.quantize(
    #         model.audio_encoder.decoder,
    #         calibration_dataset=nncf.Dataset(audio_decoder_calibration_data),
    #         preset=QuantizationPreset.MIXED,
    #         subset_size=len(audio_decoder_calibration_data),
    #         ignored_scope=None,
    #         advanced_parameters=nncf.AdvancedQuantizationParameters(
    #             overflow_fix=OverflowFix.DISABLE,
    #             disable_channel_alignment=False
    #         )
    #     )
    #     save_model(compressed_audio_decoder, compressed_audio_decoder_path)
    # model.audio_encoder.decoder = core.read_model(compressed_audio_decoder_path)
    # model.audio_encoder.decoder_compiled = core.compile_model(model.audio_encoder.decoder, DEVICE)

    #
    # Quantize audio decoder through ONNX
    #
    # compressed_audio_decoder_path = save_dir / "audio_decoder.xml"
    # if not compressed_audio_decoder_path.exists():
    #     audio_decoder_calibration_data_onnx = [dict(output_ids=it.numpy()) for it in audio_decoder_calibration_data]
    #     compressed_audio_decoder_onnx = nncf.quantize(
    #         onnx.load(audio_decoder_onnx_path),
    #         calibration_dataset=nncf.Dataset(audio_decoder_calibration_data_onnx),
    #         preset=QuantizationPreset.MIXED,
    #         subset_size=len(audio_decoder_calibration_data_onnx),
    #         ignored_scope=None,
    #         advanced_parameters=nncf.AdvancedQuantizationParameters(
    #             overflow_fix=OverflowFix.DISABLE,
    #             disable_channel_alignment=False
    #         )
    #     )
    #     onnx.save(compressed_audio_decoder_onnx, compressed_audio_decoder_path.with_suffix(".onnx"))
    #     compressed_audio_decoder_ov = convert_model(compressed_audio_decoder_path.with_suffix(".onnx"))
    #     save_model(compressed_audio_decoder_ov, compressed_audio_decoder_path)
    # model.audio_encoder.decoder = core.read_model(compressed_audio_decoder_path)
    # model.audio_encoder.decoder_compiled = core.compile_model(model.audio_encoder.decoder, DEVICE)

    return model


model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small", torchscript=True, return_dict=False).to("cpu").eval()
# print("tokens:", SAMPLE_LENGTH * model.config.audio_encoder.frame_rate + 3)

# infer_model(model, "80s pop track with bassy drums and synth", "original.wav")

# convert_to_ir(model)

model = convert_to_ov_model(model)

# Total: 31 sec
# Decoder: 26 sec (N_TOKENS - 1 calls)
# infer_model(model, "80s pop track with bassy drums and synth", f"openvino_{SAMPLE_LENGTH}.wav")

save_dir = Path("quantized/4_6/mixed_sq.15")
model = quantize(model,
                 save_dir=save_dir,
                 n_calibration_samples=6,
                 preset=QuantizationPreset.MIXED,
                 sq_alpha=0.15)

# PTQ:
# Total: 25 sec
# Decoder: 20 sec (N_TOKENS - 1 calls)
# WC:
# Total: 30
# Decoder: 25
infer_model(model, "80s pop track with bassy drums and synth", save_dir / "result_4.wav")
