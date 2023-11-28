import atexit
from threading import Thread
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


def run_upcast():
    model = partially_upcast_nodes_to_fp32.partially_upcast_nodes_to_fp32(
        ov_model.model, get_inputs_for_calibration("<human>: Which lakes are near Munich?\n<bot>:"),
        batch_size=100, verbose=True)


if __name__ == '__main__':
    model_id = "red-pajama-3b-chat"
    # models_dir = Path("c:/Users/nsavelye/workspace/projects/openvino_notebooks/notebooks/254-llm-chatbot")
    models_dir = Path("../")
    # model_dir = models_dir / model_id / "FP16"
    model_dir = models_dir / model_id / "INT8_compressed_weights"
    model_configuration = SUPPORTED_MODELS[model_id]

    core = ov.Core()
    # device = "CPU"
    device = "GPU"

    tok = AutoTokenizer.from_pretrained(models_dir / model_id / "RedPajama-INCITE-Chat-3B-v1", trust_remote_code=True)

    ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
    ov_model = OVModelForCausalLM.from_pretrained(model_dir, device=device, ov_config=ov_config,
                                                  config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
                                                  trust_remote_code=True)

    # prompt = "Which lakes are near Munich?"
    # generation_kwargs = dict(
    #     max_new_tokens=50,
    #     temperature=0.1,
    #     do_sample=0.1 > 0.0,
    #     top_p=1.0,
    #     top_k=50,
    #     repetition_penalty=1.1
    # )
    # print(run_generate(ov_model, prompt, **generation_kwargs))
    #
    # for text in run_generate(ov_model, tok, prompt, model_configuration, **generation_kwargs):
    #     print(text)

    run_upcast()
