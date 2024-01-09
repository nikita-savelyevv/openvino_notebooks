import sys
import subprocess
import json
import re

sys.path.append("/home/guest/nsavelye/workspace/openvino.genai/llm_bench/python")

config = [
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat-original/FP16", "CPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat-original/FP16", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat-original/INT8_compressed_weights", "CPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat-original/INT8_compressed_weights", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/FP16/r10", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/FP16/r20", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/FP16/r30", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/FP16/r50", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/FP16/r70", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/FP16/r100", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/INT8_compressed_weights/r10", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/INT8_compressed_weights/r20", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/INT8_compressed_weights/r30", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/INT8_compressed_weights/r50", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/INT8_compressed_weights/r70", "GPU"),
    ("/home/guest/nsavelye/workspace/fp16_calibration/notebooks/254-llm-chatbot/red-pajama-3b-chat/INT8_compressed_weights/r100", "GPU"),
]


command_template = '/home/guest/nsavelye/venvs/fp16_calibration/bin/python ' \
                   '/home/guest/nsavelye/workspace/openvino.genai/llm_bench/python/benchmark.py -m {model_path} ' \
                   '-d {device} -p "What is openvino?" -ic 32 -n 5 -rj {report_path}'
for model_path, device in config:
    report_path = f"{model_path}/perf_report_{device}.json"
    command = command_template.replace(
        "{model_path}", f"{model_path}").replace(
        "{device}", device).replace(
        "{report_path}", report_path)

    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    # Print the result
    print("Exit Code:", result.returncode)
    print("Standard Output:", result.stdout)
    print("Standard Error:", result.stderr)

    with open(report_path, "r") as f:
        report_data = json.load(f)

    average_latency = re.search(r"\[Average\] Latency: (\d+\.\d+) ms/token", result.stdout).group(1)
    report_data["average_latency"] = float(average_latency)

    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=4, ensure_ascii=False)
    # break
