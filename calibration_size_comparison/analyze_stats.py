import pickle
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from calibration_size_comparison.plot import plot_from_data


with open("../notebooks/267-distil-whisper-asr/metrics/small.en/common_voice_13_0/test-size1000_decoder-only.json",
          "r") as f:
    distil_whisper_small_decoder_only = json.load(f)
_, xs, ys = plot_from_data(distil_whisper_small_decoder_only, "Dec", "accuracy")
plt.cla()

X = None
legend = []
for size, acc in zip(xs, ys):
    with open(f"../notebooks/267-distil-whisper-asr/ptq_stats/stats_size{size}.pkl", "rb") as f:
        stats = pickle.load(f)

    x = []
    for node_name, dct in stats.items():
        for k, stat_dict in dct.items():
            if k == 'MMQ':
                for stat_name in ["min_values", "max_values"]:
                    stat_value = stat_dict[stat_name]
                    stat_value_min = stat_value.min()
                    # stat_value_max = stat_value.max()
                    # stat_value_diff = stat_value_max - stat_value_min
                    # stat_value_mean = stat_value.mean()
                    # stat_value_std = stat_value.std()
                    x.extend([
                        stat_value_min,
                        # stat_value_max,
                        # stat_value_diff,
                        # stat_value_mean,
                        # stat_value_std
                    ])
                    if X is None:
                        legend.extend([
                            f"{node_name}_{stat_name}_min",
                            # f"{node_name}_{stat_name}_max",
                            # f"{node_name}_{stat_name}_diff",
                            # f"{node_name}_{stat_name}_mean",
                            # f"{node_name}_{stat_name}_std",
                        ])
    x = np.array(x, dtype=np.float32)
    if not isinstance(X, np.ndarray):
        X = x[None]
    else:
        X = np.append(X, x[None], axis=0)

# min/max/mean/std indices
# ind = 228
# ind = 249
# ind = 568
# ind = 569
# ind = 570

# min/max/diff
# ind = 171
# ind = 187
# ind = 426
# ind = 427

# ind = 142
# plt.scatter(X[:, ind], ys)
# plt.xlabel(legend[ind])
# plt.ylabel('Accuracy')
# plt.grid()
# plt.show()
# exit(0)

reg = make_pipeline(StandardScaler(),
                    SGDRegressor(max_iter=1000, tol=1e-5, verbose=1, eta0=1e-3))
reg.fit(X, ys)

coef = reg[1].coef_
plt.bar(np.arange(len(coef)), coef)
plt.grid()
plt.show()
