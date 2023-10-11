import time
from pathlib import Path

import torch
import openvino as ov
from scipy.io import wavfile

from bark.generation import load_model, codec_decode, _flatten_codebooks
from bark import SAMPLE_RATE
import nncf

from utils import TextEncoderModel, CoarseEncoderModel, FineModel, OVBarkTextEncoder, OVBarkFineEncoder, OVBarkEncoder,\
    generate_audio, forward_data_collection

core = ov.Core()

DEVICE = "CPU"


models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

text_use_small = True
text_model_suffix = "_small" if text_use_small else ""
text_model_dir = models_dir / f"text_encoder{text_model_suffix}"
text_model_dir.mkdir(exist_ok=True)
text_encoder_path1 = text_model_dir / "bark_text_encoder_1.xml"
text_encoder_path0 = text_model_dir / "bark_text_encoder_0.xml"
quantized_text_encoder_path1 = text_model_dir / "bark_text_encoder_1_q.xml"
compressed_text_encoder_path1 = text_model_dir / "bark_text_encoder_1_c.xml"

coarse_use_small = True
coarse_model_suffix = "_small" if coarse_use_small else ""
coarse_model_dir = models_dir / f"coarse{coarse_model_suffix}"
coarse_model_dir.mkdir(exist_ok=True)
coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder.xml"
quantized_coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder_q.xml"
compressed_coarse_encoder_path = coarse_model_dir / "bark_coarse_encoder_c.xml"

fine_use_small = False
fine_model_suffix = "_small" if fine_use_small else ""
fine_model_dir = models_dir / f"fine_model{fine_model_suffix}"
fine_model_dir.mkdir(exist_ok=True)
fine_feature_extractor_path = fine_model_dir / "bark_fine_feature_extractor.xml"


def create_text_encoder():
    text_encoder = load_model(
        model_type="text", use_gpu=False, use_small=text_use_small, force_reload=False
    )

    text_encoder_model = text_encoder["model"]
    tokenizer = text_encoder["tokenizer"]

    if not text_encoder_path0.exists() or not text_encoder_path1.exists():
        text_encoder_exportable = TextEncoderModel(text_encoder_model)
        ov_model = ov.convert_model(
            text_encoder_exportable, example_input=torch.ones((1, 513), dtype=torch.int64)
        )
        ov.save_model(ov_model, text_encoder_path0)
        logits, kv_cache = text_encoder_exportable(torch.ones((1, 513), dtype=torch.int64))
        ov_model = ov.convert_model(
            text_encoder_exportable,
            example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
        )
        ov.save_model(ov_model, text_encoder_path1)
        del ov_model
        del text_encoder_exportable
    del text_encoder_model, text_encoder

    return tokenizer


def create_coarse_model():
    coarse_model = load_model(
        model_type="coarse", use_gpu=False, use_small=coarse_use_small, force_reload=False,
    )

    if not coarse_encoder_path.exists():
        coarse_encoder_exportable = CoarseEncoderModel(coarse_model)
        logits, kv_cache = coarse_encoder_exportable(
            torch.ones((1, 886), dtype=torch.int64)
        )
        ov_model = ov.convert_model(
            coarse_encoder_exportable,
            example_input=(torch.ones((1, 1), dtype=torch.int64), kv_cache),
        )
        ov.save_model(ov_model, coarse_encoder_path)
        del ov_model
        del coarse_encoder_exportable
    del coarse_model


def create_fine_model():
    fine_model = load_model(model_type="fine", use_gpu=False, use_small=fine_use_small, force_reload=False)

    if not fine_feature_extractor_path.exists():
        lm_heads = fine_model.lm_heads
        fine_feature_extractor = FineModel(fine_model)
        feature_extractor_out = fine_feature_extractor(
            3, torch.zeros((1, 1024, 8), dtype=torch.int32)
        )
        ov_model = ov.convert_model(
            fine_feature_extractor,
            example_input=(
                torch.ones(1, dtype=torch.long),
                torch.zeros((1, 1024, 8), dtype=torch.long),
            ),
        )
        ov.save_model(ov_model, fine_feature_extractor_path)
        for i, lm_head in enumerate(lm_heads):
            ov.save_model(
                ov.convert_model(lm_head, example_input=feature_extractor_out),
                fine_model_dir / f"bark_fine_lm_{i}.xml",
            )


text = "Hello, my name is Suno. And, uh — and I like banana and apples. [laughs] But I also have other interests " \
       "such as playing tic tac toe."
# text = "Hello, my name is Suno."


tokenizer = create_text_encoder()
create_coarse_model()
create_fine_model()


# ov_text_model = OVBarkTextEncoder(core, DEVICE, text_encoder_path0, text_encoder_path1)
# ov_text_model = OVBarkTextEncoder(core, DEVICE, text_encoder_path0, quantized_text_encoder_path1)
ov_text_model = OVBarkTextEncoder(core, DEVICE, text_encoder_path0, compressed_text_encoder_path1)
# ov_coarse_model = OVBarkEncoder(core, DEVICE, coarse_encoder_path)
# ov_coarse_model = OVBarkEncoder(core, DEVICE, quantized_coarse_encoder_path)
ov_coarse_model = OVBarkEncoder(core, DEVICE, compressed_coarse_encoder_path)
ov_fine_model = OVBarkFineEncoder(core, DEVICE, fine_model_dir)

# input_names = [inp.any_name for inp in ov_text_model.model2.inputs][1:]
# ov_text_model_shape = "idx:inputs/idx," + ','.join([f"{name}:inputs/{name}" for name in input_names])
# print(ov_text_model_shape)

# input_names = [inp.any_name for inp in ov_coarse_model.model.inputs][1:]
# # ov_coarse_model_shape = "idx[1,1]," + ','.join([f"{name}[1,12,257,64]" for name in input_names])
# # ov_coarse_model_shape = "idx:inputs/idx.npy," + ','.join([f"{name}:inputs/{name}.npy" for name in input_names])
# ov_coarse_model_shape = "idx:inputs2/idx," + ','.join([f"{name}:inputs2/{name}" for name in input_names])
# print(ov_coarse_model_shape)
# exit(0)

torch.manual_seed(42)
t0 = time.time()
# with forward_data_collection():
audio_array = generate_audio(tokenizer, ov_text_model, ov_coarse_model, ov_fine_model, text)
generation_duration_s = time.time() - t0
audio_duration_s = audio_array.shape[0] / SAMPLE_RATE
wavfile.write("audio.wav", SAMPLE_RATE, audio_array)

print(f"took {generation_duration_s:.0f}s to generate {audio_duration_s:.0f}s of audio")

print(f"OV text model time 1: {ov_text_model.total_time1} 2: {ov_text_model.total_time2}")
print(f"OV coarse model time: {ov_coarse_model.total_time}")
print(f"OV fine model feature extractor time: {ov_fine_model.total_time_feats} "
      f"lm heads: {ov_fine_model.total_time_lm_heads}")


# Compression

# ov_coarse_model_compressed = nncf.compress_weights(ov_coarse_model.model)
# ov.save_model(ov_coarse_model_compressed, compressed_coarse_encoder_path)
#
# ov_text_model_compressed = nncf.compress_weights(ov_text_model.model2)
# ov.save_model(ov_text_model_compressed, compressed_text_encoder_path1)

# Quantization

# ov_coarse_model_quantized = nncf.quantize(
#     ov_coarse_model.model,
#     calibration_dataset=nncf.Dataset(ov_coarse_model.forward_data),
#     preset=nncf.QuantizationPreset.MIXED,
#     subset_size=len(ov_coarse_model.forward_data),
#     fast_bias_correction=True,
#     model_type=nncf.ModelType.TRANSFORMER)
#
# ov.save_model(ov_coarse_model_quantized, quantized_coarse_encoder_path)

# ov_text_model_quantized = nncf.quantize(
#     ov_text_model.model2,
#     calibration_dataset=nncf.Dataset(ov_text_model.forward_data),
#     preset=nncf.QuantizationPreset.MIXED,
#     subset_size=len(ov_text_model.forward_data),
#     fast_bias_correction=True,
#     model_type=nncf.ModelType.TRANSFORMER)
#
# ov.save_model(ov_text_model_quantized, quantized_text_encoder_path1)
