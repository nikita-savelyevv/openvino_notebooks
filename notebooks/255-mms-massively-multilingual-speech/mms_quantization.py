import json
import sys
from contextlib import contextmanager
from itertools import islice
from pathlib import Path
from datetime import datetime

from jiwer import wer_standardize, wer
from scipy.io.wavfile import write as write_wav
import soundfile as sf

import nncf
import numpy as np
import torch
import openvino as ov

from datasets import load_dataset
from tqdm import tqdm
from transformers import Wav2Vec2ForSequenceClassification, AutoFeatureExtractor
from transformers import Wav2Vec2ForCTC, AutoProcessor


SAMPLE_LANG = ['german', 'dutch', 'french', 'spanish', 'italian', 'portuguese', 'polish', 'english'][-1]
LANG_ID = {'german': 'deu', 'french': 'fra', 'english': 'eng'}[SAMPLE_LANG]
MAX_SEQ_LENGTH = 30480

model_lid_id = "facebook/mms-lid-126"
lid_processor = AutoFeatureExtractor.from_pretrained(model_lid_id)
lid_model = Wav2Vec2ForSequenceClassification.from_pretrained(model_lid_id)

model_asr_id = "facebook/mms-1b-all"
# model_asr_id = "facebook/mms-300m"
asr_processor = AutoProcessor.from_pretrained(model_asr_id)
asr_model = Wav2Vec2ForCTC.from_pretrained(model_asr_id)
asr_processor.tokenizer.set_target_lang(LANG_ID)
asr_model.load_adapter(LANG_ID)

core = ov.Core()
device = "CPU"

lid_model_xml_path = Path('models/ov_lid_model.xml')
compressed_lid_model_xml_path = Path('models/ov_lid_model_c.xml')
quantized_lid_model_xml_path = Path('models/ov_lid_model_quantized.xml')
# quantized_lid_model_xml_path = Path('models/ov_lid_model_q-p.xml')
asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model.xml')
compressed_asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model_c.xml')
quantized_asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model_quantized.xml')
# quantized_asr_model_xml_path = Path(f'models/ov_asr_{LANG_ID}_model_q-p.xml')

# mls = load_dataset("facebook/multilingual_librispeech", SAMPLE_LANG, split="test", streaming=True)


def detect_lang(compiled_lid_model, audio_data):
    inputs = lid_processor(audio_data, sampling_rate=16_000, return_tensors="pt")

    save_input_path = Path("models/inputs/input_lid.npy")
    if not save_input_path.exists():
        np.save(save_input_path, inputs['input_values'].numpy())
    outputs = compiled_lid_model(inputs['input_values'])[0]

    lang_id = torch.argmax(torch.from_numpy(outputs), dim=-1)[0].item()
    detected_lang = lid_model.config.id2label[lang_id]

    return detected_lang


def recognize_audio(compiled_asr_model, src_audio):
    inputs = asr_processor(src_audio, sampling_rate=16_000, return_tensors="pt")

    # save_input_path = Path("models/inputs/input_asr.npy")
    # if not save_input_path.exists():
    #     np.save(save_input_path, inputs['input_values'].numpy())
    # print(inputs['input_values'].shape)
    outputs = compiled_asr_model(inputs['input_values'])[0]

    ids = torch.argmax(torch.from_numpy(outputs), dim=-1)[0]
    transcription = asr_processor.decode(ids)

    return transcription


def get_lid_model(model_path):
    input_values = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    # attention_mask = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.int32)

    if not model_path.exists() and model_path == lid_model_xml_path:
        lid_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model = ov.convert_model(lid_model, example_input={'input_values': input_values})
        ov.save_model(converted_model, lid_model_xml_path)
    compiled_lid_model = core.compile_model(model_path, device_name=device)
    return compiled_lid_model


def get_asr_model(model_path):
    input_values = torch.zeros([1, MAX_SEQ_LENGTH], dtype=torch.float)
    if not model_path.exists() and model_path == asr_model_xml_path:
        asr_model_xml_path.parent.mkdir(parents=True, exist_ok=True)
        converted_model = ov.convert_model(asr_model, example_input={'input_values': input_values})
        ov.save_model(converted_model, asr_model_xml_path)
    compiled_asr_model = core.compile_model(model_path, device_name=device)
    return compiled_asr_model


def resample(audio, src_sample_rate, dst_sample_rate):
    """
    Resample audio to specific sample rate

    Parameters:
      audio: input audio signal
      src_sample_rate: source audio sample rate
      dst_sample_rate: destination audio sample rate
    Returns:
      resampled_audio: input audio signal resampled with dst_sample_rate
    """
    if src_sample_rate == dst_sample_rate:
        return audio
    duration = audio.shape[0] / src_sample_rate
    resampled_data = np.zeros(shape=(int(duration * dst_sample_rate)), dtype=np.float32)
    x_old = np.linspace(0, duration, audio.shape[0], dtype=np.float32)
    x_new = np.linspace(0, duration, resampled_data.shape[0], dtype=np.float32)
    resampled_audio = np.interp(x_new, x_old, audio)
    return resampled_audio.astype(np.float32)


def validate(model, test_samples):
    ground_truths = []
    predictions = []
    inference_time = []
    for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
        audio_array = data_item["audio"]["array"]
        audio_array = resample(audio_array, data_item["audio"]["sampling_rate"], 16000)

        start_time = datetime.now()
        transcription = recognize_audio(model, audio_array)
        end_time = datetime.now()
        delta_time = (end_time - start_time).total_seconds()

        # print()
        # print(data_item["text"])
        # print(transcription[0])
        ground_truths.append(data_item.get("text", data_item["sentence"]))
        predictions.append(transcription)
        inference_time.append(delta_time)

    word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                             hypothesis_transform=wer_standardize)) * 100
    mean_inference_time = np.mean(inference_time)
    return mean_inference_time, word_accuracy


# print(core.read_model(lid_model_xml_path).inputs)
# print(core.read_model(asr_model_xml_path).inputs)
# exit(0)

# mls = iter(mls)  # make it iterable
# example = next(mls)  # get one example
# sf.write("french.wav", example['audio']['array'], 16000)

# compiled_lid_model = get_lid_model(lid_model_xml_path)
# compiled_lid_model = get_lid_model(compressed_lid_model_xml_path)
# compiled_lid_model = get_lid_model(quantized_lid_model_xml_path)
# start_time = datetime.now()
# lang_id = detect_lang(compiled_lid_model, example['audio']['array'])
# print(f"Language detection: {(datetime.now() - start_time).total_seconds()}")
# print(lang_id, LANG_ID)
# compiled_asr_model = get_asr_model(asr_model_xml_path)
# compiled_asr_model = get_asr_model(compressed_asr_model_xml_path)
# inputs = asr_processor(example['audio']['array'], sampling_rate=16_000, return_tensors="pt")
# print(inputs['input_values'].shape)
# exit(0)
# compiled_asr_model = get_asr_model(quantized_asr_model_xml_path)
# start_time = datetime.now()
# transcription = recognize_audio(compiled_asr_model, example['audio']['array'])
# print(f"Speech recognition: {(datetime.now() - start_time).total_seconds()}")
# print(example["text"])
# print(transcription)


# compressed_lid_model = nncf.compress_weights(core.read_model(lid_model_xml_path))
# ov.save_model(compressed_lid_model, compressed_lid_model_xml_path)
# compressed_asr_model = nncf.compress_weights(core.read_model(asr_model_xml_path))
# ov.save_model(compressed_asr_model, compressed_asr_model_xml_path)

# calibration_data = []
# for i in range(1):
#     data = asr_processor(next(mls)['audio']['array'], sampling_rate=16_000, return_tensors="pt")["input_values"]
#     calibration_data.append(data)
#
# quantized_lid_model = nncf.quantize(
#     core.read_model(lid_model_xml_path),
#     calibration_dataset=nncf.Dataset(calibration_data),
#     # preset=nncf.QuantizationPreset.MIXED,
#     preset=nncf.QuantizationPreset.PERFORMANCE,
#     subset_size=len(calibration_data),
#     fast_bias_correction=True,
#     model_type=nncf.ModelType.TRANSFORMER
# )
# ov.save_model(quantized_lid_model, quantized_lid_model_xml_path)
#
# quantized_asr_model = nncf.quantize(
#     core.read_model(asr_model_xml_path),
#     calibration_dataset=nncf.Dataset(calibration_data),
#     # preset=nncf.QuantizationPreset.MIXED,
#     preset=nncf.QuantizationPreset.PERFORMANCE,
#     subset_size=len(calibration_data),
#     fast_bias_correction=True,
#     model_type=nncf.ModelType.TRANSFORMER
# )
# ov.save_model(quantized_asr_model, quantized_asr_model_xml_path)


test_dataset_size = 1000
dataset_label = "mozilla-foundation/common_voice_13_0"
test_dataset = load_dataset(dataset_label, "en", split="test")
test_dataset = test_dataset.shuffle(seed=42)
test_samples = [sample for sample in islice(test_dataset, test_dataset_size)]

calibration_dataset = load_dataset("librispeech_asr", "clean", split="validation")#.shuffle(seed=42)

fp32_asr_model = core.compile_model(asr_model_xml_path)


save_dir = Path("metrics") / dataset_label.split('/')[1]
metrics_per_size = []
for calibration_dataset_size in (
        list(range(1, 100 + 1, 5)) +
        list(range(150, 1000 + 1, 50))
):
    calibration_data = []
    for data_item in islice(calibration_dataset, calibration_dataset_size):
        data = asr_processor(data_item['audio']['array'], sampling_rate=16_000, return_tensors="pt")["input_values"]
        calibration_data.append(data)

    quantized_asr_model = nncf.quantize(
        core.read_model(asr_model_xml_path),
        calibration_dataset=nncf.Dataset(calibration_data),
        preset=nncf.QuantizationPreset.MIXED,
        # preset=nncf.QuantizationPreset.PERFORMANCE,
        subset_size=len(calibration_data),
        fast_bias_correction=True,
        model_type=nncf.ModelType.TRANSFORMER
    )
    quantized_asr_model = core.compile_model(quantized_asr_model)

    # n_samples = 1
    # predict(ov_model, n_samples=n_samples, print_predictions=bool(0))
    # predict(quantized_ov_model, n_samples=n_samples, print_predictions=bool(0))

    transcription_time_int8, accuracy_int8 = validate(quantized_asr_model, test_samples)
    metrics_dict = {
        "calibration_dataset_size": calibration_dataset_size,
        "time_int8": transcription_time_int8,
        "accuracy_int8": accuracy_int8
    }
    if calibration_dataset_size == 1:
        transcription_time_fp32, accuracy_fp32 = validate(fp32_asr_model, test_samples)
        metrics_dict["time_fp32"] = transcription_time_fp32
        metrics_dict["accuracy_fp32"] = accuracy_fp32
    print(f"\nSize: {calibration_dataset_size}. Metrics: {metrics_dict}\n")
    metrics_per_size.append(metrics_dict)

    save_dir.mkdir(exist_ok=True, parents=True)
    with open(save_dir / "without-calibration-shuffle.json", "w") as f:
        json.dump(metrics_per_size, f, indent=4)
