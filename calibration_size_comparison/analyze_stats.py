import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration_size_comparison.plot import plot_from_data


# stats_filepath_template = "../notebooks/267-distil-whisper-asr/ptq_stats/stats_size{}.pkl"
stats_filepath_template = "../notebooks/267-distil-whisper-asr/ptq_stats_w-noop_fp32/stats_size{}.pkl"


def statistics_regression(xs, ys):
    X = None
    legend = []
    for size, acc in zip(xs, ys):
        with open(stats_filepath_template.format(size), "rb") as f:
            stats = pickle.load(f)

        x = []
        for node_name, dct in stats.items():
            for k, stat_dict in dct.items():
                if k == 'MMQ':
                    # for stat_name in ["min_values", "max_values"]:
                    #     stat_value = stat_dict[stat_name]
                    #     # stat_value_min = stat_value.min()
                    #     # stat_value_max = stat_value.max()
                    #     # stat_value_diff = stat_value_max - stat_value_min
                    #     stat_value_mean = stat_value.mean()
                    #     # stat_value_std = stat_value.std()
                    #     x.extend([
                    #         # stat_value_min,
                    #         # stat_value_max,
                    #         # stat_value_diff,
                    #         stat_value_mean,
                    #         # stat_value_std
                    #     ])
                    #     if X is None:
                    #         legend.extend([
                    #             # f"{node_name}_{stat_name}_min",
                    #             # f"{node_name}_{stat_name}_max",
                    #             # f"{node_name}_{stat_name}_diff",
                    #             f"{node_name}_{stat_name}_mean",
                    #             # f"{node_name}_{stat_name}_std",
                    #         ])

                    min_mean = stat_dict["min_values"].mean()
                    max_mean = stat_dict["max_values"].mean()
                    diff = max_mean - min_mean
                    x.extend([
                        diff
                    ])
                    if X is None:
                        legend.extend([
                            f"{node_name}_diff",
                        ])
        x = np.array(x, dtype=np.float32)
        if not isinstance(X, np.ndarray):
            X = x[None]
        else:
            X = np.append(X, x[None], axis=0)


    # ind = 0
    # ind = 18
    # ind = 19
    ind = 31
    # ind = 48
    # ind = 70
    # ind = 71
    # ind = 72
    plt.scatter(X[:, ind], ys)
    print(legend[ind])
    plt.xlabel(legend[ind])
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
    exit(0)

    reg = make_pipeline(StandardScaler(),
                        SGDRegressor(max_iter=1000, tol=1e-5, verbose=1, eta0=1e-3))
    reg.fit(X, ys)

    coef = reg[1].coef_
    plt.bar(np.arange(len(coef)), coef)
    for i, v in enumerate(coef):
        print(f"{i}, {v:.4f} {legend[i]}")
    plt.grid()
    plt.show()


def plot_statistics_distribution(xs, ys, target_node_name):
    for size, acc in zip(xs, ys):
        with open(stats_filepath_template.format(size), "rb") as f:
            stats = pickle.load(f)

        for k, stat_dict in stats[target_node_name].items():
            if k == 'MMQ':
                plt.cla()
                for stat_name in ["min_values", "max_values"]:
                    vals, bins, _ = plt.hist(stat_dict[stat_name], bins=50, label=f"{stat_name}", alpha=0.5)
                    plt.vlines(stat_dict[stat_name].mean(), min(vals), max(vals), linestyles='dashed', color='red',
                               label=f"{stat_name}_mean")
                plt.grid()
                plt.title(f"node={target_node_name}; size={size}")
                plt.legend()
                plt.show()


def plot_statistics_per_size(xs, ys, target_node_name):
    min_mean = []
    max_mean = []
    value_mean = []
    for size, acc in zip(xs, ys):
        with open(stats_filepath_template.format(size), "rb") as f:
            stats = pickle.load(f)

        for k, stat_dict in stats[target_node_name].items():
            if k == 'MMQ':
                # min_mean.append(stat_dict["min_values"].mean())
                # max_mean.append(stat_dict["max_values"].mean())
                value_mean.append(stat_dict["no_op"][0].mean())

    # plt.plot(xs, min_mean, label='min_values mean')
    # plt.plot(xs, max_mean, label='max_values mean')
    plt.plot(xs, value_mean, label='weight mean')
    plt.title(target_node_name)
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_statistics_diff_matrix(xs, ys, target_node_name):
    diffs = []
    for size1, _ in zip(xs, ys):
        diffs.append([])
        with open(stats_filepath_template.format(size1), "rb") as f:
            stats1 = pickle.load(f)
        x1 = stats1[target_node_name]['MMQ']['no_op'][0]
        for size2, _ in zip(xs, ys):
            with open(stats_filepath_template.format(size2), "rb") as f:
                stats2 = pickle.load(f)
            x2 = stats2[target_node_name]['MMQ']['no_op'][0]
            diffs[-1].append(f"{np.abs(x2 - x1).mean():.6f}")

    for row in diffs:
        print(row)


if __name__ == '__main__':
    with open("../notebooks/267-distil-whisper-asr/metrics/small.en/common_voice_13_0/test-size1000_decoder-only.json",
              "r") as f:
        distil_whisper_small_decoder_only = json.load(f)
    _, xs, ys = plot_from_data(distil_whisper_small_decoder_only, "Dec", "accuracy")
    plt.cla()

    # xs, ys = xs[:4], ys[:4]

    # statistics_regression(xs, ys)

    target_node_name = '__module.model.model.decoder.layers.3.fc2/aten::linear/MatMul_820'
    # plot_statistics_distribution(xs, ys, target_node_name)
    # plot_statistics_per_size(xs, ys, target_node_name)
    plot_statistics_diff_matrix(xs, ys, target_node_name)
