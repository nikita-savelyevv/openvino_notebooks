import atexit
import pickle
import shutil
from threading import Thread

import numpy as np

from utils import SUPPORTED_MODELS
from transformers import AutoConfig, TextIteratorStreamer
from optimum.intel.openvino import OVModelForCausalLM
from pathlib import Path
import openvino as ov
from openvino.runtime import Tensor
from transformers import AutoTokenizer
import partially_upcast_nodes_to_fp32


def run_generate(ov_model, tok, text, tokenizer_kwargs=None, **generation_kwargs):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    input_tokens = tok(text, return_tensors="pt", **tokenizer_kwargs)
    input_ids = input_tokens.input_ids

    streamer = TextIteratorStreamer(tok)
    generation_kwargs = dict(input_ids=input_ids, streamer=streamer, **generation_kwargs)
    thread = Thread(target=ov_model.generate, kwargs=generation_kwargs)
    thread.start()
    atexit.register(thread.join)
    for new_text in streamer:
        yield new_text

    # answer = ov_model.generate(input_ids=input_ids, **generation_kwargs)
    # return tok.batch_decode(answer)[0]


def get_inputs_for_calibration(ov_model, tok, example_string, tokenizer_kwargs=None):
    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}
    inputs = dict(tok(example_string, return_tensors="np", **tokenizer_kwargs))
    for input_name in ov_model.key_value_input_names:
        model_inputs = ov_model.model.input(input_name)
        shape = model_inputs.get_partial_shape()
        shape[0] = inputs['input_ids'].shape[0]
        if shape[2].is_dynamic:
            shape[2] = 0
        if shape[1].is_dynamic:
            shape[1] = 0
        inputs[input_name] = Tensor(model_inputs.get_element_type(), shape.get_shape())
    return inputs


if __name__ == '__main__':
    #
    # Model loading
    #

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}

    core = ov.Core()

    models_dir = Path("./models")

    # MODEL_ID = "red-pajama-3b-chat"
    # MODEL_ID = "T5"
    # MODEL_ID = "tiny-sd-unet"
    # MODEL_ID = "codegen-2B-multi"
    MODEL_ID = "gpt-neox-20b"

    if MODEL_ID in ["red-pajama-3b-chat", "tiny-sd-unet", "T5"]:
        half_type = "f16"
        model_dir = models_dir / MODEL_ID / "FP16"
        # model_dir = models_dir / MODEL_ID / "FP16_calibrated"
        # model_dir = models_dir / MODEL_ID / "INT8_compressed_weights"
        device = "GPU"
        # device = "CPU"

        if MODEL_ID == "red-pajama-3b-chat":
            example_prompt = "<human>: Which lakes are near Munich?\\n<bot>:"
        elif MODEL_ID == "T5":
            example_prompt = "ultra close color photo portrait of rainbow owl with deer horns in the woods"
        elif MODEL_ID == "tiny-sd-unet":
            with open("unet_example_input.pkl", "rb") as f:
                unet_example_input = pickle.load(f)
        else:
            raise Exception("Unknown model")
    elif MODEL_ID in ["codegen-2B-multi", "gpt-neox-20b"]:
        half_type = "bf16"
        device = "CPU"
        # ov_config["INFERENCE_PRECISION_HINT"] = "f32"     # otherwise BF16 is used
        if MODEL_ID == "codegen-2B-multi":
            model_dir = Path(
                "/home/devuser/nsavelye/workspace/openvino.genai/llm_bench/python/codegen-2B-multi/pytorch/dldt/FP32")
            example_prompt = "# this function implement Fourier transform for imput array X"
        elif MODEL_ID == "gpt-neox-20b":
            model_dir = Path(
                "/home/devuser/nsavelye/workspace/openvino.genai/llm_bench/python/gpt-neox-20b/fp16/pytorch/dldt/FP32")
            example_prompt = "Which lakes are near Munich?"
        else:
            raise Exception("Unknown model")
    else:
        raise Exception("Unknown model")

    if MODEL_ID in ["red-pajama-3b-chat", "codegen-2B-multi", "gpt-neox-20b"]:
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ov_model_for_causal_lm = OVModelForCausalLM.from_pretrained(
            model_dir, device=device, ov_config=ov_config,
            config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True), trust_remote_code=True)
        model = ov_model_for_causal_lm.model
    elif MODEL_ID == "T5":
        model = core.read_model(model_dir / "encoder_ir.xml")
    elif MODEL_ID == "tiny-sd-unet":
        model = core.read_model(model_dir / "unet.xml")
    else:
        raise Exception("Unknown model")

    #
    # Upcasting
    #

    SAVE_MODEL = bool(0)

    if MODEL_ID in ["red-pajama-3b-chat", "codegen-2B-multi", "gpt-neox-20b"]:
        batch_size = -1
        example_input = get_inputs_for_calibration(ov_model_for_causal_lm, tok, example_prompt)
        if MODEL_ID in ["codegen-2B-multi", "gpt-neox-20b"]:
            position_ids = np.cumsum(example_input["attention_mask"], axis=1) - 1
            position_ids[example_input["attention_mask"] == 0] = 1
            example_input["position_ids"] = position_ids
    elif MODEL_ID == "T5":
        batch_size = -1
        # from diffusers import DiffusionPipeline
        # tokenizer = DiffusionPipeline.from_pretrained("DeepFloyd/IF-I-M-v1.0").tokenizer
        tokenizer = AutoTokenizer.from_pretrained(models_dir / MODEL_ID / "tokenizer")
        example_input = tokenizer(example_prompt, max_length=77, padding="max_length", return_tensors="np").input_ids
    elif MODEL_ID == "tiny-sd-unet":
        batch_size = -1
        example_input = unet_example_input
    else:
        raise Exception("Unknown model")

    # shape_str = ""
    # for k, v in example_input.items():
    #     file_path = f"ex_in/{k}.npy"
    #     data = v if isinstance(v, np.ndarray) else v.data
    #     np.save(file_path, data.astype(np.float32) if data.dtype == np.float16 else data)
    #     shape_str += f"{k}:{file_path},"
    #     # shape_str += f"{k}{list(v.shape)},".replace(' ', '')
    # print(shape_str)
    # exit(0)

    # upcasted_model = model_upcast_utils.partially_upcast_nodes_to_fp32(model, example_input)
    upcast_ratio = 1.0
    upcasted_model = partially_upcast_nodes_to_fp32.partially_upcast_nodes_to_fp32(
        model, example_input, batch_size=batch_size, verbose=True, half_type=half_type, upcast_ratio=upcast_ratio,
        operation_types=['MatMul', 'Convolution', 'Softmax', 'MVN', 'Multiply', 'Divide', 'Add', 'Subtract', 'Concat',
                         'Power', 'Transpose', 'Broadcast', 'ShapeOf']
    )
    # upcasted_model = model

    if SAVE_MODEL:
        calibrated_model_dir = Path(f"{model_dir}_calibrated_{upcast_ratio:.2f}")
        if MODEL_ID in ["red-pajama-3b-chat", "codegen-2B-multi", "gpt-neox-20b"]:
            # shutil.copytree(model_dir, calibrated_model_dir)
            ov.save_model(upcasted_model, calibrated_model_dir / "openvino_model.xml", compress_to_fp16=False)
            for filename in ["config.json", "added_tokens.json", "special_tokens_map.json", "tokenizer.json",
                             "tokenizer_config.json", "vocab.json"]:
                if (model_dir / filename).exists():
                    shutil.copy(str(model_dir / filename), str(calibrated_model_dir / filename))
        elif MODEL_ID == "T5":
            ov.save_model(upcasted_model, calibrated_model_dir / "encoder_ir.xml", compress_to_fp16=True)
        elif MODEL_ID == "tiny-sd-unet":
            ov.save_model(upcasted_model, calibrated_model_dir / "unet.xml")
        else:
            raise Exception("Unknown model")
