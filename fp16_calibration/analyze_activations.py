import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.colors as colors


def get_sym_log_bins(data, n_bins, decrease_min_coef=1.0, increase_max_coef=1.0):
    min_value, max_value = data.min(), data.max()
    min_value *= increase_max_coef if min_value < 0 else decrease_min_coef
    max_value *= increase_max_coef if max_value > 0 else decrease_min_coef
    linthresh = int(
        max(1, np.ceil(-np.log10(max(1e-10, min(abs(min_value), abs(max_value), np.abs(data).min()))))))
    if min_value < 0:
        if max_value > 0:
            negative_share = linthresh + np.log10(-min_value)
            positive_share = linthresh + np.log10(max_value)
            n_negative_bins = int(n_bins * negative_share / (negative_share + positive_share))
            n_positive_bins = int(n_bins * positive_share / (negative_share + positive_share))
            bins = (-np.logspace(np.log10(-min_value), -linthresh, n_negative_bins)).tolist()[:-1] + \
                   np.logspace(-linthresh, np.log10(max_value), n_positive_bins).tolist()
        else:
            bins = -np.logspace(np.log10(-min_value), np.log10(-max_value), n_bins)
    else:
        bins = np.logspace(np.log10(min_value), np.log10(max_value), n_bins)
    return bins, linthresh


def plot_distr(fp16_act, fp32_act, large_error_ind, num_large_error_elements, outcome2, node_name, filepath):
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    n_bins = 100
    for i in range(2):
        for j in range(2):
            label = "FP16" if i == 0 else "FP32"
            data_all = fp16_act if i == 0 else fp32_act
            data_plot = data_all[large_error_ind].ravel() if j == 0 else data_all.ravel()
            if j == 0 and num_large_error_elements == 0:
                continue

            bins, linthresh = get_sym_log_bins(data_plot, n_bins, decrease_min_coef=0.9, increase_max_coef=1.1)
            try:
                axs[i, j].hist(data_plot, bins=bins)
            except Exception as e:
                print(bins)
                raise e
            axs[i, j].set_xscale('symlog', linthresh=10 ** -linthresh)
            axs[i, j].set_yscale('log')
            axs[i, j].set_title(f'{label} {"large errors" if j == 0 else "all"}. '
                                f'min:{data_plot.min():.5f} max:{data_plot.max():.5f}')
            axs[i, j].grid()

    plt.suptitle(f"{node_name} {outcome2}")
    # plt.show()
    plt.savefig(f"distribution_images/{filepath.stem.replace('%', '_')}.png")
    plt.close(fig)


def plot_activation_matrix(fp16_act, fp32_act, large_error_mask, outcome2, node_name, filepath):
    fp16_act_2d = np.mean(fp16_act, axis=0)
    fp32_act_2d = np.mean(fp32_act, axis=0)
    if len(fp16_act_2d.shape) == 3:
        fp16_act_2d = np.mean(fp16_act_2d, axis=0)
        fp32_act_2d = np.mean(fp32_act_2d, axis=0)

    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    # fig, axs = plt.subplots(1, 1, figsize=(8, 6))
    # axs = [[None, None], [axs, None]]
    for i in range(2):
        for j in range(2):
            if i == 0:
                data = fp16_act_2d if j == 0 else fp32_act_2d
                label = "FP16" if j == 0 else "FP32"
                # continue
            else:
                mask_2d = large_error_mask.astype(int)
                mask_2d = np.mean(mask_2d, axis=0)
                if len(mask_2d.shape) == 3:
                    mask_2d = np.mean(mask_2d, axis=0)
                if j == 0:
                    data = mask_2d
                    label = "Large errors"
                else:
                    continue
                    data = np.ones_like(fp32_act_2d, np.float32) * np.mean(fp32_act_2d)
                    data[np.where(mask_2d)] = fp32_act_2d[mask_2d]
                    label = "Large errors values"
            linthresh = int(max(1, np.ceil(-np.log10(max(1e-10, np.abs(data).min())))))
            im = axs[i][j].imshow(data, interpolation="nearest",
                               #norm=colors.SymLogNorm(vmin=data.min(), vmax=data.max(), linthresh=linthresh)
                               )
            cbar = axs[i][j].figure.colorbar(im, ax=axs[i][j], fraction=0.046)
            axs[i][j].set_title(label)
            # axs[i].axis('off')
            axs[i][j].set_aspect(data.shape[1] / data.shape[0])

    plt.suptitle(f"{node_name} {outcome2}")
    plt.tight_layout()
    # plt.show()
    plt.savefig(f"activations_images/{filepath.stem.replace('%', '_')}.png")


def compute_sqnr(x, y):
    # x -- original, y -- quantized
    Ps = np.linalg.norm(x)
    Pn = np.nan_to_num(np.linalg.norm(x - y), posinf=np.finfo(np.float32).max)
    return 20 * np.log10(Ps / Pn)


# model_id = "red-pajama-3b-chat"
# model_id = "tiny-sd-unet"
# model_id = "T5"
model_id = "codegen-2B-multi"

activations_dir = Path("activations/")
folder_path = Path("activations") / model_id

thresholds_per_type = {
    'Convolution': (0.1, 0.003, 0.00),
    'MatMul': (0.1, 0.04, 0.03),
}


columns = ["Node", "Shape", "Outcome 1", "Outcome 2",
           "SQNR", "rel error (100 - t1)% quantile",
           "FP16\nlarge error\nelements abs mean", "FP16\nfull abs mean",
           "FP32\nlarge error\nelements abs mean", "FP32\nfull abs mean",
           ]
df = pd.DataFrame(columns=columns)

sqnrs = []
sqnrs_upcasted = []
sqnrs_not_upcasted = []

sqnrs_matmul_upcasted = []
sqnrs_matmul_not_upcasted = []
sqnrs_conv_upcasted = []
sqnrs_conv_not_upcasted = []

for filepath in tqdm(sorted(folder_path.glob('*'))):
    node_name = str(filepath.name).replace('%', '/')[:-13]

    thresholds = (None, None, None)
    node_type = None
    for k in thresholds_per_type.keys():
        if k in node_name:
            node_type = k
            break
    if node_type is None:
        if "Multiply" in node_name:
            node_type = "MatMul"
    thresholds = thresholds_per_type[node_type]

    filepath_str = str(filepath)
    if "fp32" in filepath_str:
        continue

    fp16_act = np.nan_to_num(np.load(filepath), posinf=np.finfo(np.float16).max)
    fp32_act = np.load(filepath_str.replace("fp16", "fp32"))

    outcome1 = int(filepath_str[-7])
    outcome2 = int(filepath_str[-5])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rel_error = np.abs(2 * (fp16_act - fp32_act) / (np.abs(fp16_act) + np.abs(fp32_act)))
    mean_rel_error = np.mean(rel_error)
    if outcome1 != int(not(mean_rel_error < thresholds[2])):
        print(f"Warning stored and computed outcome1 do not align "
              f"{node_name}, {outcome1}, {mean_rel_error}, {thresholds[2]}")

    large_error_mask = rel_error >= thresholds[0]
    num_large_error_elements = np.sum(large_error_mask)

    large_error_ind = np.where(large_error_mask)
    fp16_large_error_mean = 0 if num_large_error_elements == 0 else np.mean(np.abs(fp16_act[large_error_ind]))
    fp32_large_error_mean = 0 if num_large_error_elements == 0 else np.mean(np.abs(fp32_act[large_error_ind]))

    rel_diff_ratio = num_large_error_elements / np.prod(rel_error.shape)
    assert outcome2 == int(rel_diff_ratio > thresholds[1]), f"{node_name}, {outcome2}, {rel_diff_ratio}, {thresholds[1]}"

    sqnr = compute_sqnr(fp32_act, fp16_act)
    quantile = np.quantile(rel_error, q=1 - thresholds[1])

    df.loc[len(df)] = [node_name, str(fp32_act.shape), outcome1, outcome2,
                       sqnr, quantile,
                       fp16_large_error_mean, np.mean(np.abs(fp16_act)),
                       fp32_large_error_mean, np.mean(np.abs(fp32_act)),
                       ]

    sqnrs.append(sqnr)
    if outcome2:
        sqnrs_upcasted.append(sqnr)
        if node_type == "MatMul":
            sqnrs_matmul_upcasted.append(sqnr)
        else:
            sqnrs_conv_upcasted.append(sqnr)
    else:
        sqnrs_not_upcasted.append(sqnr)
        if node_type == "MatMul":
            sqnrs_matmul_not_upcasted.append(sqnr)
        else:
            sqnrs_conv_not_upcasted.append(sqnr)

    # if outcome2 == 0:
    #     continue
    # plot_distr(fp16_act, fp32_act, large_error_ind, num_large_error_elements, outcome2, node_name, filepath)
    # plot_activation_matrix(fp16_act, fp32_act, large_error_mask, outcome2, node_name, filepath)



# print("DataFrame:")
# print(df)

# df.to_excel(f"output_{model_id}.xlsx", index=False)

sqnrs = np.array(sqnrs)
# bins, linthresh = get_sym_log_bins(sqnrs, 100, decrease_min_coef=0.9, increase_max_coef=1.1)
# plt.hist(sqnrs, bins)
# plt.xscale('symlog', linthresh=10 ** -linthresh)
# plt.yscale('log')
# plt.grid()

data_splits = [(sqnrs_upcasted, sqnrs_not_upcasted)]
data_labels = ["All ops"]

if model_id == "tiny-sd-unet":
    data_splits.extend([
        (sqnrs_matmul_upcasted, sqnrs_matmul_not_upcasted),
        (sqnrs_conv_upcasted, sqnrs_conv_not_upcasted)
    ])
    data_labels.extend(["MatMul ops", "Conv ops"])

for (sqnrs_upcasted, sqnrs_not_upcasted), data_label in zip(data_splits, data_labels):
    min_threshold = 0
    too_low_sqnrs = sqnrs < min_threshold
    print("Excluding sqnrs from histogram:", sqnrs[np.where(too_low_sqnrs)])
    sqnrs = sqnrs[np.where(~too_low_sqnrs)]
    # plt.hist(sqnrs, np.linspace(sqnrs.min(), sqnrs.max(), 50))
    # plt.hist([sqnrs_upcasted, sqnrs_not_upcasted], np.linspace(sqnrs.min(), sqnrs.max(), 50), stacked=True, alpha=0.5)
    plt.hist(sqnrs_not_upcasted, np.linspace(sqnrs.min(), sqnrs.max(), 50), alpha=0.5, label="Not upcasted")
    plt.hist(sqnrs_upcasted, np.linspace(sqnrs.min(), sqnrs.max(), 50), alpha=0.5, label="Upcasted")
    plt.title(f"{model_id} ({data_label})")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.yscale("log")
    # plt.show()
    save_file_name = f"sqnrs_{model_id}.png" if model_id != "tiny-sd-unet" else f"sqnrs_{model_id}_{data_label}.png"
    plt.savefig(save_file_name)
    plt.cla()
