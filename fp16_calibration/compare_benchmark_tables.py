import pandas as pd
from pathlib import Path


def fill_str(s, max_len):
    return s + ' ' * max(max_len - len(s), 0)


table1_path = Path("C:\\Users\\nsavelye\\workspace\\models\\T5-encoder\\FP16\\benchmark_average_counters_report_fp16.csv")
table2_path = Path("C:\\Users\\nsavelye\\workspace\\models\\T5-encoder\\FP16_calibrated_0.30\\benchmark_average_counters_report_fp16_c0.30.csv")

df1 = pd.read_csv(table1_path, delimiter=';')
df2 = pd.read_csv(table2_path, delimiter=';')

data1, data2 = {}, {}
for df, data in zip([df1, df2], [data1, data2]):
    for i, it in df.iterrows():
        if it["layerName"] != "Total":
            data[it["layerName"]] = (
                it["execStatus"], it["layerType"], it["execType"], it["realTime (ms)"], it["cpuTime (ms)"])
        else:
            data[it["layerName"]] = ("", "", "", it["realTime (ms)"], it["cpuTime (ms)"])

df = pd.DataFrame(columns=["layerName",
                           "execStatus1", "layerType1", "execType1", "realTime1", "cpuTime1",
                           "execStatus2", "layerType2", "execType2", "realTime2", "cpuTime2",
                           ])

max_node_name_len = 100
max_exec_status_len = 21
max_layer_type_len = 15
max_exec_type_len = 35
for k in set(list(data1.keys()) + list(data2.keys())):
    row_dict = {"layerName": k}
    for i, data_i in zip([1, 2], [data1, data2]):
        if k in data_i:
            d = data_i[k]
        else:
            d = ("-", "-", "-", 1, 1)
        row_dict[f"execStatus{i}"] = d[0]
        row_dict[f"layerType{i}"] = d[1]
        row_dict[f"execType{i}"] = d[2]
        row_dict[f"realTime{i}"] = d[3]
        row_dict[f"cpuTime{i}"] = d[4]
    df.loc[len(df)] = row_dict

df.to_excel("compare_reports.xlsx", index=False)
