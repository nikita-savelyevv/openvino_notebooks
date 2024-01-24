import io
import shutil
from contextlib import contextmanager
from datetime import datetime
from itertools import islice
from typing import Any, List
import json
import tempfile

import numpy as np
import openvino as ov
from pathlib import Path
from datasets import load_dataset
from openvino import Tensor
from optimum.intel.openvino import OVModelForSpeechSeq2Seq
from scipy.io import wavfile
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

from jiwer import wer, wer_standardize

import nncf

core = ov.Core()

device = "CPU"

# model_size_id = "large-v2"
model_size_id = "small.en"
model_id = f"distil-whisper/distil-{model_size_id}"
model_dir = Path(model_id.split("/")[-1])
quantized_model_dir = model_dir / "quantized"

processor = AutoProcessor.from_pretrained(model_id)


def convert_to_ov():
    if not model_dir.exists():
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(
            model_id, export=True, compile=False
        )
        ov_model.half()
        # ov_model.generation_config = pt_distil_model.generation_config
        ov_model.save_pretrained(model_dir)
    else:
        ov_model = OVModelForSpeechSeq2Seq.from_pretrained(model_dir, compile=False)
    return ov_model


class InferRequestWrapper:
    def __init__(self, request, data_cache):
        self.request = request
        self.data_cache = data_cache

    def __call__(self, *args, **kwargs):
        self.data_cache.append(*args)
        return self.request(*args, **kwargs)

    def infer(self, inputs: Any = None, share_inputs: bool = False):
        self.data_cache.append(inputs)
        return self.request.infer(inputs, share_inputs)

    def start_async(
            self,
            inputs: Any = None,
            userdata: Any = None,
            share_inputs: bool = False,
    ):
        self.data_cache.append(inputs)
        self.request.infer(inputs, share_inputs)

    def wait(self):
        pass

    def get_tensor(self, name: str):
        return Tensor(self.request.results[name])

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self.request, attr)


def extract_input_features(sample=None, audio_array=None):
    if audio_array is None:
        audio_array = sample["audio"]["array"]
        audio_array = resample(audio_array, sample["audio"]["sampling_rate"], 16000)
    input_features = processor(
        audio_array,
        sampling_rate=16000,
        return_tensors="pt",
    ).input_features
    return input_features


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


def time_it(obj, fn_name, time_list):
    original_fn = getattr(obj, fn_name)

    def wrapper(*args, **kwargs):
        start_time = datetime.now()
        result = original_fn(*args, **kwargs)
        end_time = datetime.now()
        time_list.append((end_time - start_time).total_seconds())
        return result

    setattr(obj, fn_name, wrapper)


def collect_calibration_dataset(ov_model, calibration_dataset_size):
    # Overwrite model request properties, saving the original ones for restoring later
    original_encoder_request = ov_model.encoder.request
    original_decoder_with_past_request = ov_model.decoder_with_past.request
    encoder_calibration_data = []
    decoder_calibration_data = []
    ov_model.encoder.request = InferRequestWrapper(original_encoder_request, encoder_calibration_data)
    ov_model.decoder_with_past.request = InferRequestWrapper(original_decoder_with_past_request,
                                                             decoder_calibration_data)

    # calibration_dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    calibration_dataset = load_dataset("librispeech_asr", "clean", split="validation").shuffle(seed=42)
    for sample in tqdm(islice(calibration_dataset, calibration_dataset_size), desc="Collecting calibration data",
                       total=calibration_dataset_size):
        input_features = extract_input_features(sample)
        ov_model.generate(input_features)

    ov_model.encoder.request = original_encoder_request
    ov_model.decoder_with_past.request = original_decoder_with_past_request

    return encoder_calibration_data, decoder_calibration_data


def quantize(ov_model, calibration_dataset_size, encoder_sq_alpha, decoder_sq_alpha, cleanup_model=False):
    # encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(ov_model,
    #                                                                                  calibration_dataset_size)
    # print(len(encoder_calibration_data), len(decoder_calibration_data))

    # save_dir_name = f"subset{calibration_dataset_size}_enc-sq-{encoder_sq_alpha:.2f}_dec-sq-{decoder_sq_alpha:.2f}"
    # save_dir = quantized_model_dir / save_dir_name
    save_dir = Path(tempfile.TemporaryDirectory().name)
    if not save_dir.exists():
        encoder_calibration_data, decoder_calibration_data = collect_calibration_dataset(ov_model,
                                                                                         calibration_dataset_size)
        print("Quantizing encoder")
        quantized_encoder = nncf.quantize(
            ov_model.encoder.model,
            nncf.Dataset(encoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(encoder_calibration_data),
            fast_bias_correction=True,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=encoder_sq_alpha)
        )
        ov.save_model(quantized_encoder, save_dir / "openvino_encoder_model.xml")

        print("Quantizing decoder with past")
        quantized_decoder_with_past = nncf.quantize(
            ov_model.decoder_with_past.model,
            nncf.Dataset(decoder_calibration_data),
            preset=nncf.QuantizationPreset.MIXED,
            subset_size=len(decoder_calibration_data),
            fast_bias_correction=True,
            model_type=nncf.ModelType.TRANSFORMER,
            advanced_parameters=nncf.AdvancedQuantizationParameters(smooth_quant_alpha=decoder_sq_alpha)
        )
        ov.save_model(quantized_decoder_with_past, save_dir / "openvino_decoder_with_past_model.xml")

        shutil.copy(model_dir / "config.json", save_dir / "config.json")
        shutil.copy(model_dir / "openvino_decoder_model.xml", save_dir / "openvino_decoder_model.xml")
        shutil.copy(model_dir / "openvino_decoder_model.bin", save_dir / "openvino_decoder_model.bin")

    quantized_ov_model = OVModelForSpeechSeq2Seq.from_pretrained(save_dir, compile=False)
    quantized_ov_model.to(device)
    quantized_ov_model.compile()

    if cleanup_model:
        shutil.rmtree(str(save_dir))

    return quantized_ov_model


def predict(ov_model, n_samples, print_predictions):
    # dataset = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    dataset = load_dataset("librispeech_asr", "clean", split="test", streaming=True).take(n_samples)

    whole_infer_times = []
    encoder_infer_times = []
    decoder_infer_times = []
    decoder_with_past_infer_times = []
    time_it(ov_model, "generate", whole_infer_times)
    time_it(ov_model.encoder, "forward", encoder_infer_times)
    time_it(ov_model.decoder, "forward", decoder_infer_times)
    time_it(ov_model.decoder_with_past, "forward", decoder_with_past_infer_times)

    for sample in tqdm(islice(dataset, n_samples), desc="Running", disable=print_predictions,
                       total=n_samples):
        input_features = extract_input_features(sample)
        predicted_ids = ov_model.generate(input_features)
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        if print_predictions:
            print()
            print(f"Reference: {sample['text']}")
            print(f"Result: {transcription[0]}")

    print()
    print(f"Whole inference time. Mean: {np.mean(whole_infer_times):.3f} s. "
          f"Sum: {np.sum(whole_infer_times):.3f} s. Count: {len(whole_infer_times)} calls")
    print(f"Encoder inference time: Mean: {np.mean(encoder_infer_times):.3f} s. "
          f"Sum: {np.sum(encoder_infer_times):.3f} s. Count: {len(encoder_infer_times)} calls")
    print(f"Decoder inference time: Mean: {np.mean(decoder_infer_times):.3f} s. "
          f"Sum: {np.sum(decoder_infer_times):.3f} s. Count: {len(decoder_infer_times)} calls")
    print(f"Decoder with past inference time: "
          f"Mean: {np.mean(decoder_with_past_infer_times):.3f} s. Sum: {np.sum(decoder_with_past_infer_times):.3f} s. "
          f"Count: {len(decoder_with_past_infer_times)} calls")


def validate(ov_model, test_samples):
    ground_truths = []
    predictions = []
    inference_time = []
    for data_item in tqdm(test_samples, desc="Measuring performance and accuracy"):
        input_features = extract_input_features(data_item)

        start_time = datetime.now()
        predicted_ids = ov_model.generate(input_features)
        end_time = datetime.now()
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        delta_time = (end_time - start_time).total_seconds()

        # print()
        # print(data_item["text"])
        # print(transcription[0])
        ground_truths.append(data_item.get("text", data_item["sentence"]))
        predictions.append(transcription[0])
        inference_time.append(delta_time)

    word_accuracy = (1 - wer(ground_truths, predictions, reference_transform=wer_standardize,
                             hypothesis_transform=wer_standardize)) * 100
    mean_inference_time = np.mean(inference_time)
    return mean_inference_time, word_accuracy


def read_audio(audio_path):
    def audio_to_float(audio):
        """
        convert audio signal to floating point format
        """
        return audio.astype(np.float32) / np.iinfo(audio.dtype).max

    sample_rate, audio = wavfile.read(
        io.BytesIO(open(audio_path, 'rb').read()))
    audio = audio_to_float(audio)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = resample(audio, sample_rate, 16000)
    return audio


ov_model = convert_to_ov()
ov_model.to(device)
ov_model.compile()

audio = read_audio(Path("downloaded_video.wav"))
predicted_ids = ov_model.generate(extract_input_features(audio_array=audio))
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print(transcription)

test_dataset_size = 1000
dataset_label = "mozilla-foundation/common_voice_13_0"
test_dataset = load_dataset(dataset_label, "en", split="test")
test_dataset = test_dataset.shuffle(seed=42)
sliced_test_dataset = islice(test_dataset, test_dataset_size) if test_dataset_size != -1 else test_dataset
test_samples = [sample for sample in sliced_test_dataset]


save_dir = Path("metrics") / dataset_label.split('/')[1]
metrics_per_size = []
for i, calibration_dataset_size in enumerate(
        # list(range(1, 100 + 1, 1)) +
        # list(range(150, 1000 + 1, 50))
    [68, 69]
):
    quantized_ov_model = quantize(ov_model,
                                  calibration_dataset_size=calibration_dataset_size,
                                  encoder_sq_alpha=0.50,
                                  decoder_sq_alpha=0.95,
                                  cleanup_model=True)

    # audio = read_audio(Path("downloaded_video.wav"))
    # predicted_ids = quantized_ov_model.generate(extract_input_features(audio_array=audio))
    # transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
    # print(transcription)

    # n_samples = 1
    # predict(ov_model, n_samples=n_samples, print_predictions=bool(0))
    # predict(quantized_ov_model, n_samples=n_samples, print_predictions=bool(0))

    transcription_time_int8, accuracy_int8 = validate(quantized_ov_model, test_samples)
    metrics_dict = {
        "calibration_dataset_size": calibration_dataset_size,
        "time_int8": transcription_time_int8,
        "accuracy_int8": accuracy_int8
    }
    if i == 0:
        transcription_time_fp32, accuracy_fp32 = validate(ov_model, test_samples)
        metrics_dict["time_fp32"] = transcription_time_fp32
        metrics_dict["accuracy_fp32"] = accuracy_fp32
    print(f"\nSize: {calibration_dataset_size}. Metrics: {metrics_dict}\n")
    metrics_per_size.append(metrics_dict)

    save_dir.mkdir(exist_ok=True)
    with open(save_dir / f"with-calibration-shuffle_test-size{test_dataset_size}.json", "w") as f:
        json.dump(metrics_per_size, f, indent=4)
