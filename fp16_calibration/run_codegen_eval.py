from pathlib import Path
import subprocess

python_path = "/home/devuser/nsavelye/venvs/fp16_calibration/bin/python"
project_path = "/home/devuser/nsavelye/workspace/bigcode-evaluation-harness"
generate_command_template = \
    '{python_path} {project_path}/main.py --modeltype {model_type} --model {model_path} --tasks humaneval --precision '\
    '{precision}{limit_arg}{limit_start_arg} --max_length_generation 650 --temperature 0.8 --do_sample True ' \
    '--n_samples 10 --batch_size 8 --trust_remote_code --generation_only --save_generations --save_generations_path {' \
    'generations_path}'
evaluate_command_template = \
    '{python_path} {project_path}/main.py --modeltype {model_type} --model {model_path} --precision {precision}{' \
    'limit_arg}{limit_start_arg} --tasks humaneval --load_generations_path {generations_path} --allow_code_execution ' \
    '--temperature 0.8 --n_samples 10'

limit = None
limit_start = None
models_dir = Path("/home/devuser/nsavelye/workspace/openvino.genai/llm_bench/python/codegen-2B-multi/pytorch/dldt")
run_args = [
    {
        "model_type": "ov_causal",
        "model_path": str(models_dir / "FP32"),
        "precision": "bf16",
        "limit": limit,
        "limit_start": limit_start,
    },
    {
        "model_type": "ov_causal",
        "model_path": str(models_dir / "FP32"),
        "precision": "fp32",
        "limit": limit,
        "limit_start": limit_start,
    },
    # {
    #     "model_type": "causal",
    #     "model_path": str(models_dir / "FP32_pt"),
    #     "precision": "fp32",
    #     "limit": limit,
    #     "limit_start": limit_start,
    # },
    {
        "model_type": "ov_causal",
        "model_path": str(models_dir / "FP32_calibrated_0.30"),
        "precision": "bf16",
        "limit": limit,
        "limit_start": limit_start,
    },
    # {
    #     "model_type": "ov_causal",
    #     "model_path": str(models_dir / "FP32_calibrated_0.50"),
    #     "precision": "bf16",
    #     "limit": limit,
    #     "limit_start": limit_start,
    # },
    {
        "model_type": "ov_causal",
        "model_path": str(models_dir / "FP32_calibrated_0.70"),
        "precision": "bf16",
        "limit": limit,
        "limit_start": limit_start,
    },
    # {
    #     "model_type": "ov_causal",
    #     "model_path": str(models_dir / "FP32_calibrated_0.90"),
    #     "precision": "bf16",
    #     "limit": limit,
    #     "limit_start": limit_start,
    # },
    {
        "model_type": "ov_causal",
        "model_path": str(models_dir / "FP32_calibrated_1.00"),
        "precision": "bf16",
        "limit": limit,
        "limit_start": limit_start,
    },
]

for run_arg in run_args:
    model_type = run_arg["model_type"]
    model_path = run_arg["model_path"]
    limit = run_arg["limit"]
    precision = run_arg["precision"]

    limit_str = "" if limit is None else f"_{limit}"
    limit_start_str = "" if limit_start is None else f"_s{limit_start}"
    generations_path = str(f"{model_path}/generations_{precision}{limit_str}{limit_start_str}.json")
    result_path = str(f"{model_path}/result_{precision}{limit_str}{limit_start_str}.txt")

    limit_arg = f" --limit {limit}" if limit is not None else ""
    limit_start_arg = f" --limit_start {limit_start}" if limit_start is not None else ""
    generate_command = generate_command_template.\
        replace("{python_path}", python_path).\
        replace("{project_path}", project_path).\
        replace("{model_type}", model_type).\
        replace("{model_path}", model_path).\
        replace("{precision}", precision).\
        replace("{limit_arg}", limit_arg).\
        replace("{limit_start_arg}", limit_start_arg).\
        replace("{generations_path}", generations_path)
    evaluate_command = evaluate_command_template.\
        replace("{python_path}", python_path).\
        replace("{project_path}", project_path).\
        replace("{model_type}", model_type).\
        replace("{model_path}", model_path).\
        replace("{precision}", precision).\
        replace("{limit_arg}", limit_arg).\
        replace("{limit_start_arg}", limit_start_arg).\
        replace("{generations_path}", generations_path)

    print(generate_command)
    subprocess.check_output(generate_command.split(' '))
    print(evaluate_command)
    with open(result_path, "w") as f:
        f.write(subprocess.check_output(evaluate_command.split(' ')).decode("utf-8"))
