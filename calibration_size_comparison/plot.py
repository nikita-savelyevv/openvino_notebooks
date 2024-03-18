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
    xs, ys = zip(*sorted(list(zip(xs, ys)), key=lambda it: it[0]))
    plt.plot(xs, ys, label=label)
    return fp32_acc, xs, ys


def plot_distil_whisper():
    with open("../notebooks/267-distil-whisper-asr/metrics/common_voice_13_0/with-calibration-shuffle.json", "r") as f:
        distil_whisper_with_shuffle = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/common_voice_13_0/without-calibration-shuffle.json", "r") \
            as f:
        distil_whisper_without_shuffle = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/common_voice_13_0/with-calibration-shuffle_test-size-1.json",
              "r") as f:
        distil_whisper_with_shuffle_full_test = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/test-size1000.json", "r") as f:
        distil_whisper_large_old = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/small.en/common_voice_13_0/test-size1000_decoder-only.json", "r") as f:
        distil_whisper_small_decoder_only = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/small.en/common_voice_13_0/test-size1000_decoder-only_no-sq.json", "r") as f:
        distil_whisper_small_decoder_only_no_sq = json.load(f)

    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size100_decoder-only_no-sq.json", "r") as f:
        distil_whisper_large_decoder_only_no_sq = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size1000_decoder-only_no-sq.json", "r") as f:
        distil_whisper_large_decoder_only_no_sq_1000 = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size100_decoder-only_sq-0.95.json", "r") as f:
        distil_whisper_large_decoder_only = json.load(f)

    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size100_encoder-only_no-sq.json", "r") as f:
        distil_whisper_large_encoder_only_no_sq = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size100_encoder-only_sq-0.5.json", "r") as f:
        distil_whisper_large_encoder_only = json.load(f)

    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size1000_sq.json", "r") as f:
        distil_whisper_large = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size1000_no-sq.json", "r") as f:
        distil_whisper_large_no_sq = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/common_voice_13_0/test-size1000_sq_optimum-fix.json", "r") as f:
        distil_whisper_large_sq_optimum_fix = json.load(f)

    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/librispeech_asr/test-size1000_sq.json", "r") as f:
        distil_whisper_large_librispeech = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/librispeech_asr/test-size1000_no-sq.json", "r") as f:
        distil_whisper_large_librispeech_no_sq = json.load(f)
    with open("../notebooks/267-distil-whisper-asr/metrics/large-v2/librispeech_asr/test-size1000_sq_optimum-fix.json", "r") as f:
        distil_whisper_large_librispeech_optimum_fix = json.load(f)

    # fp32_acc, xs, _ = plot_from_data(distil_whisper_without_shuffle, "Without shuffle")
    # fp32_acc_1000, xs, _ = plot_from_data(distil_whisper_with_shuffle, "1000 samples", "accuracy")
    # fp32_acc_full, _ = plot_from_data(distil_whisper_with_shuffle_full_test, "Full dataset", "accuracy")
    # plt.hlines([fp32_acc_full], xmin=min(xs), xmax=max(xs), colors='r', label="Baseline full dataset")
    # plt.hlines([fp32_acc_1000], xmin=min(xs), xmax=max(xs), colors='orange', label="Baseline 1000 samples")

    # fp32_acc_large, xs, _ = plot_from_data(distil_whisper_large, "1000 samples", "accuracy")
    # plt.hlines([fp32_acc_large], xmin=min(xs), xmax=max(xs), colors='r', label="Baseline")

    # fp32_acc, xs, _ = plot_from_data(distil_whisper_small_decoder_only, "Quantized Decoder", "accuracy")
    # plot_from_data(distil_whisper_small_decoder_only_no_sq, "Quantized Decoder w/o SQ", "accuracy")

    # fp32_acc, xs, _ = plot_from_data(distil_whisper_large_decoder_only, "Quantized Decoder", "accuracy")
    # plot_from_data(distil_whisper_large_decoder_only_no_sq, "Quantized Decoder w/o SQ", "accuracy")

    # fp32_acc, xs, _ = plot_from_data(distil_whisper_large_encoder_only, "Quantized Encoder", "accuracy")
    # plot_from_data(distil_whisper_large_encoder_only_no_sq, "Quantized Encoder w/o SQ", "accuracy")

    # fp32_acc_1000, xs, _ = plot_from_data(distil_whisper_large_decoder_only_no_sq_1000, "Quantized Decoder w/o SQ 1000", "accuracy")
    # fp32_acc_100, _, _ = plot_from_data(distil_whisper_large_decoder_only_no_sq, "Quantized Decoder w/o SQ 100", "accuracy")
    # plt.hlines([fp32_acc_1000], xmin=min(xs), xmax=max(xs), colors='r', label="Baseline")

    plot_from_data(distil_whisper_large, "Quantized w. SQ", "accuracy")
    plot_from_data(distil_whisper_large_sq_optimum_fix, "Quantized w. SQ (optimum fix)", "accuracy")
    fp32_acc, xs, _ = plot_from_data(distil_whisper_large_no_sq, "Quantized w/o SQ", "accuracy")

    # plot_from_data(distil_whisper_large_librispeech, "Quantized w. SQ", "accuracy")
    # plot_from_data(distil_whisper_large_librispeech_optimum_fix, "Quantized w. SQ (optimum fix)", "accuracy")
    # fp32_acc, xs, _ = plot_from_data(distil_whisper_large_librispeech_no_sq, "Quantized w/o SQ", "accuracy")

    plt.hlines([fp32_acc], xmin=min(xs), xmax=max(xs), colors='r', label="Baseline")

    plt.ylabel("Accuracy on common_voice_13 (1000 samples)")
    # plt.ylabel("Accuracy on librispeech (1000 samples)")
    plt.xlabel("Calibration dataset size")
    plt.title("Distil-Whisper large-v2")
    plt.tight_layout()
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

    # fp32_acc_core, xs, _ = plot_from_data(core_values, "Core i9", "top1")
    # fp32_acc_spr, _, _ = plot_from_data(spr_values, "SPR", "top1")
    # plt.hlines([fp32_acc_core], xmin=min(xs), xmax=max(xs), colors='C5', label="Baseline on Core i9")
    # plt.hlines([fp32_acc_spr], xmin=min(xs), xmax=max(xs), colors='C6', label="Baseline on SPR")
    # plt.ylabel("Top-1 accuracy on ImageNet (1000 images)")
    # plt.xlabel("Calibration dataset size")
    # plt.legend()
    # plt.grid()
    # plt.show()

    fp32_acc, xs, _ = plot_from_data(spr_values, "Quantized", "top1")
    plt.hlines([fp32_acc], xmin=min(xs), xmax=max(xs), colors='C3', label="Baseline")
    plt.ylabel("Top-1 accuracy on ImageNet (1000 images)")
    plt.xlabel("Calibration dataset size")
    plt.title("CLIP Image Classification")
    plt.legend()
    plt.grid()
    plt.show()


def plot_grammar_correction():
    with open("../notebooks/214-grammar-correction/metrics/flan-t5-large-grammar-synthesis/test_748"
              "/metrics_2024-01-26 19-14-38.json", "r") as f:
        values = json.load(f)
    with open("../notebooks/214-grammar-correction/metrics/flan-t5-large-grammar-synthesis/test_748"
              "/metrics_2024-03-01 13-15-29_optimum-fix.json", "r") as f:
        values_optimum_fix = json.load(f)

    # with open("../notebooks/214-grammar-correction/metrics/grammar-synthesis-small/test_748"
    #           "/metrics_2024-01-24 14-34-28.json", "r") as f:
    #     values = json.load(f)
    # with open("../notebooks/214-grammar-correction/metrics/grammar-synthesis-small/test_748"
    #           "/metrics_2024-03-01 10-10-07_optimum-fix.json", "r") as f:
    #     values_optimum_fix = json.load(f)

    # with open("../notebooks/214-grammar-correction/metrics/grammar-synthesis-small/test_748"
    #           "/metrics_2024-01-25 21-52-48.json", "r") as f:
    #     values_no_quantile = json.load(f)

    fp32_acc, xs, _ = plot_from_data(values, "Quantized", "accuracy")
    plot_from_data(values_optimum_fix, "Quantized (optimum fix)", "accuracy")
    # plot_from_data(values_no_quantile, "Quantized (no quantile)", "accuracy")
    plt.hlines([fp32_acc], xmin=min(xs), xmax=max(xs), colors='C3', label="Baseline")
    plt.ylabel("Accuracy (748 samples)")
    plt.xlabel("Calibration dataset size")
    plt.title("Grammar Correction (Large)")
    plt.tight_layout()
    plt.legend()
    plt.grid()
    plt.show()


if __name__ == '__main__':
    # plot_distil_whisper()
    # plot_clip()
    plot_grammar_correction()
