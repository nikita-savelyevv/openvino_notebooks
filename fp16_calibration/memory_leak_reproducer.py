from transformers import AutoConfig, AutoTokenizer
from optimum.intel.openvino import OVModelForCausalLM
from pathlib import Path
import openvino as ov
import psutil
from datetime import datetime
import gc


def red_pijama_partial_text_processor(partial_text, new_text):
    if new_text == '<':
        return partial_text

    partial_text += new_text
    return partial_text.split('<bot>:')[-1]


def get_allocated_memory():
    return (psutil.virtual_memory().total - psutil.virtual_memory().available) / 2**30


model_id = "red-pajama-3b-chat"
models_dir = Path("models")
model_dir = models_dir / model_id / "FP16"
model_configuration = {"model_id": "togethercomputer/RedPajama-INCITE-Chat-3B-v1", "start_message": "",
                       "history_template": "\n<human>:{user}\n<bot>:{assistant}", "stop_tokens": [29, 0],
                       "partial_text_processor": red_pijama_partial_text_processor,
                       "current_message_template": "\n<human>:{user}\n<bot>:{assistant}"}

core = ov.Core()
device = "GPU"

if not model_dir.exists():
    ov_model = OVModelForCausalLM.from_pretrained(model_configuration["model_id"], export=True, compile=False,
                                                  cache_dir="./cache")
    ov_model.half()
    ov_model.save_pretrained(model_dir)
    del ov_model
ov_config = {'PERFORMANCE_HINT': 'LATENCY', 'NUM_STREAMS': '1', "CACHE_DIR": ""}
ov_model = OVModelForCausalLM.from_pretrained(model_dir, device=device, ov_config=ov_config,
                                              config=AutoConfig.from_pretrained(model_dir, trust_remote_code=True),
                                              trust_remote_code=True)

print(f"Starting. Total allocated memory: {get_allocated_memory():.2f} GB.")
start_time = datetime.now()
for i in range(100):
    core.compile_model(ov_model.model, device)
    gc.collect()
    print(f"Iter {i} time {(datetime.now() - start_time).total_seconds():.2f} sec. "
          f"Total allocated memory: {get_allocated_memory():.2f} GB.")
    start_time = datetime.now()
