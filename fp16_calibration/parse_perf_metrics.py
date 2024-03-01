import json
from pathlib import Path
import pprint

models_dir = Path("/dev/data/nsavelye/workspace/openvino.genai/llm_bench/python/gpt-neox-20b/fp16/pytorch/dldt")
report_paths = [
    models_dir / "FP32/report_fp32.json",
    models_dir / "FP32/report_bf16.json",
    models_dir / "FP16/report_fp32.json",
    models_dir / "FP16/report_bf16.json",
    # models_dir / "stateful/FP32/report_fp32.json",
    # models_dir / "stateful/FP32/report_bf16.json",
    # models_dir / "FP32_bf16inp/report_fp32.json",
    # models_dir / "FP32_bf16inp/report_bf16.json",
    # models_dir / "FP32_fp32inp/report_fp32.json",
    # models_dir / "FP32_fp32inp/report_bf16.json",
    models_dir / "INT8/report_fp32.json",
    models_dir / "INT8/report_bf16.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_1.00_only-mvn-matmul/report.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_1.00/report.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_0.90/report.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_0.80/report.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_0.70/report.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_0.60/report.json",
    models_dir / "calibration_with_special_types/matmul-mvn/calibrated_0.50/report.json",
]

for report_path in report_paths:
    with open(report_path, 'r') as f:
        data = json.load(f)
    # pprint.pprint(data)
    latencies = []
    for res in data['perfdata']['results']:
        latencies.append(res['latency'])
    print('/'.join(str(report_path).split('/')[-3:]), sum(latencies) / len(latencies))
