import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "results"

def ensure_results_folder():
    os.makedirs(RESULTS_DIR, exist_ok=True)


def save_results_csv(results, filename="final_results.csv"):
    ensure_results_folder()
    df = pd.DataFrame(results)
    output_path = os.path.join(RESULTS_DIR, filename)
    df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")


def plot_parameter_curve(parameter_results, filename="parameter_curve.png"):
    ensure_results_folder()

    df = pd.DataFrame(parameter_results)
    csv_path = os.path.join(RESULTS_DIR, "parameter_results.csv")
    df.to_csv(csv_path, index=False)

    plt.figure(figsize=(10, 6))

    for mode in df["mode"].unique():
        subset = df[df["mode"] == mode]
        plt.plot(
            subset["J"],
            subset["svm_f1"] * 100,
            marker="o",
            label=mode.upper()
        )

    plt.xlabel("Wavelet Scattering Scale Parameter J")
    plt.ylabel("SVM F1-score (%)")
    plt.title("F1-score vs Wavelet Scattering Parameter")
    plt.legend()
    plt.tight_layout()

    output_path = os.path.join(RESULTS_DIR, filename)
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved {output_path}")