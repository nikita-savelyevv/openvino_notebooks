import csv
from pathlib import Path


def read_csv_to_list_of_dicts(filename):
    data = []
    with open(filename, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile, delimiter=';')
        header = True
        for row in reader:
            data.append(row)
            if header:
                data.pop()
                header = False
    return data


models_dir = Path("/dev/data/nsavelye/workspace/openvino.genai/llm_bench/python/gpt-neox-20b/fp16/pytorch/dldt")
data1 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/all_types" / "benchmark_average_counters_report.csv")
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_broadcast" / "benchmark_average_counters_report.csv")   # ?
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_reshape" / "benchmark_average_counters_report.csv")     # ?
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_select" / "benchmark_average_counters_report.csv")      # selects do not fuse
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_softmax" / "benchmark_average_counters_report.csv")       # just softmax in fp32
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_add" / "benchmark_average_counters_report.csv")   # ?
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_mvn" / "benchmark_average_counters_report.csv")
# data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_multiply" / "benchmark_average_counters_report.csv")
data2 = read_csv_to_list_of_dicts(models_dir / "exec_type_diff_experiment/w-o_matmul" / "benchmark_average_counters_report.csv")

exec_types1, exec_types2 = {}, {}
for dct in data1:
    exec_types1[dct["layerName"]] = dct["execType"]
for dct in data2:
    exec_types2[dct["layerName"]] = dct["execType"]


for node_name, exec_type1 in exec_types1.items():
    if node_name not in exec_types2:
        print(f'Missing {node_name}')
        continue
    exec_type2 = exec_types2[node_name]
    if exec_type1 != exec_type2:
        print(node_name, exec_type1, exec_type2)

print(set(exec_types2.keys()).difference(exec_types1.keys()))