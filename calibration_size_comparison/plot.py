import json
import matplotlib.pyplot as plt


def plot_from_data(data, label, accuracy_field_name):
    xs, ys = [], []
    fp32_acc = None
    for i, d in enumerate(data):
        xs.append(d["calibration_dataset_size"])
        ys.append(d[f"{accuracy_field_name}_int8"])

        if i == 0:
            fp32_acc = d[f"{accuracy_field_name}_fp32"]
    plt.plot(xs, ys, label=label)
    return fp32_acc, xs


def plot_distil_whisper():
    with open("../notebooks/267-distil-whisper-asr/metrics/common_voice_13_0/with-calibration-shuffle.json", "r") as f:
        distil_whisper_with_shuffle = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/common_voice_13_0/without-calibration-shuffle.json", "r") \
            as f:
        distil_whisper_without_shuffle = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/common_voice_13_0/with-calibration-shuffle_test-size-1.json",
              "r") as f:
        distil_whisper_with_shuffle_full_test = json.load(f)


    # fp32_acc, xs = plot_from_data(distil_whisper_without_shuffle, "Without shuffle")
    fp32_acc_1000, xs = plot_from_data(distil_whisper_with_shuffle, "1000 samples", "accuracy")
    fp32_acc_full, _ = plot_from_data(distil_whisper_with_shuffle_full_test, "Full dataset", "accuracy")
    plt.hlines([fp32_acc_full], xmin=min(xs), xmax=max(xs), colors='r', label="Baseline full dataset")
    plt.hlines([fp32_acc_1000], xmin=min(xs), xmax=max(xs), colors='orange', label="Baseline 1000 samples")
    plt.ylabel("Accuracy on common_voice_13")
    plt.xlabel("Calibration dataset size")
    plt.legend()
    plt.grid()
    plt.show()


def plot_clip():
    with open("../notebooks/228-clip-zero-shot-image-classification/metrics/test_1000"
              "/metrics_2024-01-17 22-25-29.json", "r") as f:
        core_values = json.load(f)
    with open("../notebooks/228-clip-zero-shot-image-classification/metrics/test_1000"
              "/metrics_2024-01-17 23-03-22.json", "r") as f:
        spr_values = json.load(f)

    # fp32_acc_core, xs = plot_from_data(core_values, "Core i9", "top1")
    # fp32_acc_spr, _ = plot_from_data(spr_values, "SPR", "top1")
    # plt.hlines([fp32_acc_core], xmin=min(xs), xmax=max(xs), colors='C5', label="Baseline on Core i9")
    # plt.hlines([fp32_acc_spr], xmin=min(xs), xmax=max(xs), colors='C6', label="Baseline on SPR")
    # plt.ylabel("Top-1 accuracy on ImageNet (1000 images)")
    # plt.xlabel("Calibration dataset size")
    # plt.legend()
    # plt.grid()
    # plt.show()

    fp32_acc, xs = plot_from_data(spr_values, "Quantized", "top1")
    plt.hlines([fp32_acc], xmin=min(xs), xmax=max(xs), colors='C3', label="Baseline")
    plt.ylabel("Top-1 accuracy on ImageNet (1000 images)")
    plt.xlabel("Calibration dataset size")
    plt.title("CLIP Image Classification")
    plt.legend()
    plt.grid()
    plt.show()


def plot_grammar_correction():
    with open("../notebooks/214-grammar-correction/metrics/grammar-synthesis-small/test_748_old/"
              "metrics_2024-01-22 19-07-05.json", "r") as f:
        values = json.load(f)

    fp32_acc, xs = plot_from_data(values, "Quantized", "accuracy")
    plt.hlines([fp32_acc], xmin=min(xs), xmax=max(xs), colors='C3', label="Baseline")
    plt.ylabel("Accuracy (748 samples)")
    plt.xlabel("Calibration dataset size")
    plt.title("Grammar Correction")
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # plot_distil_whisper()
    # plot_clip()
    plot_grammar_correction()
