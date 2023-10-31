from pathlib import Path


from functools import wraps
import torch
import openvino as ov
from typing import Tuple

DEFAULT_SYSTEM_PROMPT = """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as 
possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, 
dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature. If a 
question does not make any sense, or is not factually coherent, explain why instead of answering something not 
correct. If you don't know the answer to a question, please don't share false information.\
"""


def red_pijama_partial_text_processor(partial_text, new_text):
    if new_text == '<':
        return partial_text

    partial_text += new_text
    return partial_text.split('<bot>:')[-1]


def llama_partial_text_processor(partial_text, new_text):
    new_text = new_text.replace("[INST]", "").replace("[/INST]", "")
    partial_text += new_text
    return partial_text


SUPPORTED_MODELS = {
    "red-pajama-3b-chat": {"model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1", "start_message": "",
                           "history_template": "\n<human>:{user}\n<bot>:{assistant}", "stop_tokens": [29, 0],
                           "partial_text_processor": red_pijama_partial_text_processor,
                           "current_message_template": "\n<human>:{user}\n<bot>:{assistant}"},
    "llama-2-chat-7b": {"model_id": "meta-llama/Llama-2-7b-chat-hf",
                        "start_message": f"<s>[INST] <<SYS>>\n{DEFAULT_SYSTEM_PROMPT}\n<</SYS>>\n\n",
                        "history_template": "{user}[/INST]{assistant}</s><s>[INST]",
                        "current_message_template": "{user} [/INST]{assistant}",
                        "tokenizer_kwargs": {"add_special_tokens": False},
                        "partial_text_processor": llama_partial_text_processor,
                        "revision": "5514c85fedd6c4fc0fc66fa533bc157de520da73"},
    "mpt-7b-chat": {"model_id": "mosaicml/mpt-7b-chat",
                    "start_message": f"<|im_start|>system\n {DEFAULT_SYSTEM_PROMPT}<|im_end|>",
                    "history_template": "<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}<|im_end|>",
                    "current_message_template": "\"<|im_start|>user\n{user}<im_end><|im_start|>assistant\n{assistant}",
                    "stop_tokens": ["<|im_end|>", "<|endoftext|>"]},
}


def flattenize_inputs(inputs):
    """
    Helper function for making nested inputs flattens
    """
    flatten_inputs = []
    for input_data in inputs:
        if input_data is None:
            continue
        if isinstance(input_data, (list, tuple)):
            flatten_inputs.extend(flattenize_inputs(input_data))
        else:
            flatten_inputs.append(input_data)
    return flatten_inputs


def cleanup_torchscript_cache():
    """
    Helper for removing cached model representation
    """
    torch._C._jit_clear_class_registry()
    torch.jit._recursive.concrete_type_store = torch.jit._recursive.ConcreteTypeStore()
    torch.jit._state._clear_class_state()


def convert_mpt(pt_model: torch.nn.Module, model_path: Path):
    """
    MPT model conversion function

    Params:
      pt_model: PyTorch model
      model_path: path for saving model
    Returns:
      None
    """
    ov_out_path = Path(model_path) / "openvino_model.xml"
    pt_model.config.save_pretrained(ov_out_path.parent)
    pt_model.config.use_cache = True
    outs = pt_model(input_ids=torch.ones((1, 10), dtype=torch.long),
                    attention_mask=torch.ones((1, 10), dtype=torch.long))
    inputs = ["input_ids"]
    outputs = ["logits"]

    dynamic_shapes = {"input_ids": {1: "seq_len"}, "attention_mask": {1: "seq_len"}}
    for idx in range(len(outs.past_key_values)):
        inputs.extend([f"past_key_values.{idx}.key", f"past_key_values.{idx}.value"])
        dynamic_shapes[inputs[-1]] = {2: "past_sequence + sequence"}
        dynamic_shapes[inputs[-2]] = {3: "past_sequence + sequence"}
        outputs.extend([f"present.{idx}.key", f"present.{idx}.value"])

    inputs.append("attention_mask")
    dummy_inputs = {"input_ids": torch.ones((1, 2), dtype=torch.long), "past_key_values": outs.past_key_values,
                    "attention_mask": torch.ones((1, 12), dtype=torch.long)}
    pt_model.config.torchscript = True
    orig_forward = pt_model.forward

    @wraps(orig_forward)
    def ts_patched_forward(input_ids: torch.Tensor, past_key_values: Tuple[Tuple[torch.Tensor]],
                           attention_mask: torch.Tensor):
        pkv_list = list(past_key_values)
        outs = orig_forward(input_ids=input_ids, past_key_values=pkv_list, attention_mask=attention_mask)
        return (outs.logits, tuple(outs.past_key_values))

    pt_model.forward = ts_patched_forward
    ov_model = ov.convert_model(pt_model, example_input=dummy_inputs)
    pt_model.forward = orig_forward
    for inp_name, m_input, input_data in zip(inputs, ov_model.inputs, flattenize_inputs(dummy_inputs.values())):
        input_node = m_input.get_node()
        if input_node.element_type == ov.Type.dynamic:
            m_input.get_node().set_element_type(ov.Type.f32)
        shape = list(input_data.shape)
        if inp_name in dynamic_shapes:
            for k in dynamic_shapes[inp_name]:
                shape[k] = -1
        input_node.set_partial_shape(ov.PartialShape(shape))
        m_input.get_tensor().set_names({inp_name})

    for out, out_name in zip(ov_model.outputs, outputs):
        out.get_tensor().set_names({out_name})

    ov_model.validate_nodes_and_infer_types()
    ov.save_model(ov_model, ov_out_path)
    del ov_model
    cleanup_torchscript_cache()
    del pt_model
