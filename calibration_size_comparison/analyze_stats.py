import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration_size_comparison.plot import plot_from_data


# stats_filepath_template = "../notebooks/267-distil-whisper-asr/ptq_stats/stats_size{}.pkl"
stats_filepath_template = "../notebooks/267-distil-whisper-asr/ptq_stats_w-noop_fp32/stats_size{}.pkl"
# stats_filepath_template = "../notebooks/267-distil-whisper-asr/ptq_stats_no-sq/stats_size{}.pkl"


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
                    pass
                    # if np.prod(stat_dict["min_values"].shape) > 1:
                    #     continue
                    # for stat_name in ["min_values", "max_values"]:
                    #     stat_value = stat_dict[stat_name]
                    #     # stat_value_min = stat_value.min()
                    #     # stat_value_max = stat_value.max()
                    #     stat_value_mean = stat_value.mean()
                    #     # stat_value_std = stat_value.std()
                    #     x.extend([
                    #         # stat_value_min,
                    #         # stat_value_max,
                    #         stat_value_mean,
                    #         # stat_value_std
                    #     ])
                    #     if X is None:
                    #         legend.extend([
                    #             # f"MM_{node_name}_{stat_name}_min",
                    #             # f"MM_{node_name}_{stat_name}_max",
                    #             f"MM_{node_name}_{stat_name}_mean",
                    #             # f"MM_{node_name}_{stat_name}_std",
                    #         ])
                    #
                    # # min_mean = stat_dict["min_values"].mean()
                    # # max_mean = stat_dict["max_values"].mean()
                    # # diff = max_mean - min_mean
                    # # x.extend([
                    # #     diff
                    # # ])
                    # # if X is None:
                    # #     legend.extend([
                    # #         f"MM_{node_name}_diff",
                    # #     ])
                # elif k == "FBC":
                #     # for stat_name in ["mean_values_pre", "mean_values_post"]:
                #     #     stat_value = stat_dict[stat_name]
                #     #     # stat_value_min = stat_value.min()
                #     #     # stat_value_max = stat_value.max()
                #     #     stat_value_mean = stat_value.mean()
                #     #     # stat_value_std = stat_value.std()
                #     #     x.extend([
                #     #         # stat_value_min,
                #     #         # stat_value_max,
                #     #         stat_value_mean,
                #     #         # stat_value_std
                #     #     ])
                #     #     if X is None:
                #     #         legend.extend([
                #     #             # f"FBC_{node_name}_{stat_name}_min",
                #     #             # f"FBC_{node_name}_{stat_name}_max",
                #     #             f"FBC_{node_name}_{stat_name}_mean",
                #     #             # f"FBC_{node_name}_{stat_name}_std",
                #     #         ])
                #
                #     min_mean = stat_dict["mean_values_pre"].mean()
                #     max_mean = stat_dict["mean_values_post"].mean()
                #     diff = max_mean - min_mean
                #     x.extend([
                #         diff
                #     ])
                #     if X is None:
                #         legend.extend([
                #             f"FBC_{node_name}_diff",
                #         ])
                elif k == 'SQ':
                    for stat_name in ["abs_max"]:
                        stat_value = stat_dict[stat_name]
                        # stat_value_min = stat_value.min()
                        # stat_value_max = stat_value.max()
                        stat_value_mean = stat_value.mean()
                        # stat_value_std = stat_value.std()
                        x.extend([
                            # stat_value_min,
                            # stat_value_max,
                            stat_value_mean,
                            # stat_value_std
                        ])
                        if X is None:
                            legend.extend([
                                # f"SQ_{node_name}_{stat_name}_min",
                                # f"SQ_{node_name}_{stat_name}_max",
                                f"SQ_{node_name}_{stat_name}_mean",
                                # f"SQ_{node_name}_{stat_name}_std",
                            ])
        x = np.array(x, dtype=np.float32)
        if not isinstance(X, np.ndarray):
            X = x[None]
        else:
            X = np.append(X, x[None], axis=0)

    def plot_scatter(ind):
        plt.cla()
        plt.scatter(X[:, ind], ys)
        print(ind, legend[ind])
        plt.xlabel(legend[ind])
        plt.ylabel('Accuracy')
        plt.grid()
        plt.show()

    # 2, 3, 4, 5, 7, 10, 15, 27, 30, 37, 40, 47, 50
    # for ind in range(len(legend)):
    #     plot_scatter(ind)

    ind = 3
    plot_scatter(ind)
    exit(0)

    reg = make_pipeline(StandardScaler(),
                        SGDRegressor(max_iter=1000, tol=1e-5, verbose=1, eta0=1e-3))

    # reg = make_pipeline(StandardScaler(),
    #                     SGDClassifier(max_iter=1000, tol=1e-3, verbose=1, eta0=1e-1, alpha=1e-1, loss='log_loss'))
    # ys = [int(it > 76) for it in ys]
    reg.fit(X, ys)

    coef = reg[1].coef_
    # coef = coef[0]
    plt.bar(np.arange(len(coef)), coef)
    for i, v in enumerate(coef):
        print(f"{i}, {v:.4f} {legend[i]}")
    plt.grid()
    # plt.tight_layout()
    plt.xlabel("feature id")
    plt.ylabel("Corr. coefficient")
    plt.show()


def plot_statistics_distribution(xs, ys, target_node_name):
    for size, acc in zip(xs, ys):
        with open(stats_filepath_template.format(size), "rb") as f:
            stats = pickle.load(f)

        for k, stat_dict in stats[target_node_name].items():
            if k == 'MMQ':
                pass
                # plt.cla()
                # for stat_name in ["min_values", "max_values"]:
                #     vals, bins, _ = plt.hist(stat_dict[stat_name], bins=50, label=f"{stat_name}", alpha=0.5)
                #     plt.vlines(stat_dict[stat_name].mean(), min(vals), max(vals), linestyles='dashed', color='red',
                #                label=f"{stat_name}_mean")
                # plt.grid()
                # plt.title(f"node={target_node_name}; size={size}")
                # plt.legend()
                # plt.show()
            elif k == 'SQ':
                plt.cla()
                for stat_name in ["abs_max"]:
                    vals, bins, _ = plt.hist(stat_dict[stat_name].ravel(), bins=50, label=f"{stat_name}", alpha=0.5)
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
    value_max = []
    for size, acc in zip(xs, ys):
        with open(stats_filepath_template.format(size), "rb") as f:
            stats = pickle.load(f)

        for k, stat_dict in stats[target_node_name].items():
            # if k == 'MMQ':
            #     min_mean.append(stat_dict["min_values"].mean())
            #     max_mean.append(stat_dict["max_values"].mean())
            #     # value_mean.append(stat_dict["no_op"][0].mean())
            if k == 'SQ':
                value_max.append(stat_dict["abs_max"].max())

    # plt.plot(xs, min_mean, label='min_values mean')
    # plt.plot(xs, max_mean, label='max_values mean')
    # plt.plot(xs, value_mean, label='weight mean')
    plt.plot(xs, value_max, label='max stat')
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


def compare_sq_stats(xs, ys, target_node_name):
    noop_stats_path = "../notebooks/267-distil-whisper-asr/stats/small.en/decoder-sq-noop-fix/stats_size{}.pkl"
    stats_path = "../notebooks/267-distil-whisper-asr/stats/small.en/decoder-sq-fix/stats_size{}.pkl"

    max_absmax = []
    for size, acc in zip(xs, ys):
        with open(noop_stats_path.format(size), "rb") as f:
            noop_stats = pickle.load(f)
        with open(stats_path.format(size), "rb") as f:
            stats = pickle.load(f)

        stat_noop = np.array(noop_stats[target_node_name]['SQ']['abs_max'], dtype=np.float32)
        stat_aggregated = stats[target_node_name]['SQ']['abs_max']
        stat_noop_aggregated = stat_noop.max(axis=0)
        assert np.abs(stat_noop_aggregated - stat_aggregated).max() < 1e-6

        max_absmax.append(stat_aggregated.max())

        # max_per_sample = stat_noop.mean(axis=-1).ravel()
        # plt.plot(np.arange(len(max_per_sample)), max_per_sample, label=f"size={size}")

    # plt.grid()
    # plt.legend()
    # plt.show()

    plt.plot(xs, max_absmax, label='max stat')
    ys = np.array(ys) / 100
    plt.plot(xs, ys, label='Accuracy')
    plt.title(target_node_name)
    plt.grid(True)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    with open("../notebooks/267-distil-whisper-asr/metrics/small.en/common_voice_13_0/test-size1000_decoder-only.json", "r") as f:
        distil_whisper_small_decoder_only = json.load(f)
    # with open("../notebooks/267-distil-whisper-asr/metrics/small.en/common_voice_13_0/test-size1000_decoder-only_no-sq.json", "r") as f:
    #     distil_whisper_small_decoder_only = json.load(f)
    _, xs, ys = plot_from_data(distil_whisper_small_decoder_only, "Dec", "accuracy")
    plt.cla()

    xs, ys = xs[:5], ys[:5]
    # xs, ys = xs[10:100], ys[10:100]

    # statistics_regression(xs, ys)

    # target_node_name = '__module.model.model.decoder.layers.3.encoder_attn.out_proj/aten::linear/MatMul'
    target_node_name = '__module.model.model.decoder.layers.3.encoder_attn.out_proj/aten::linear/MatMul_815'
    # target_node_name = '__module.model.model.decoder.layers.3.fc2/aten::linear/MatMul_820'
    # target_node_name = '__module.model.model.decoder.layers.0.self_attn/aten::bmm/MatMul_110'
    # plot_statistics_distribution(xs, ys, target_node_name)
    # plot_statistics_per_size(xs, ys, target_node_name)
    # plot_statistics_diff_matrix(xs, ys, target_node_name)
    compare_sq_stats(xs, ys, target_node_name)