import csv
import matplotlib.pyplot as plt

SIZE_CSV = "07_gpu_memory_transfer_size_global.csv"
TIME_CSV = "06_gpu_memory_transfer_time_global.csv"


def load_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main():
    size_rows = load_csv(SIZE_CSV)
    time_rows = load_csv(TIME_CSV)

    directions = [r["direction"] for r in size_rows]
    total_mb = [float(r["total_MB"]) for r in size_rows]
    count = [int(r["count"]) for r in size_rows]

    total_ns = [float(r["total_time_ns"]) for r in time_rows]
    total_ms = [t / 1e6 for t in total_ns]
    time_percent = [float(r["time_percent"]) for r in time_rows]

    colors = ["#4C72B0", "#DD8452"]  # bleu = H2D, orange = D2H

    # === SEULEMENT 2 GRAPHIQUES MAINTENANT ===
    fig, axes = plt.subplots(1, 2, figsize=(6, 5))

    # --- 1. Volume total transféré par direction ---
    ax1 = axes[0]
    bars1 = ax1.bar(directions, total_mb, color=colors)
    ax1.set_yscale("log")
    ax1.set_ylabel("Volume total transféré (Mo, échelle log)")
    ax1.set_title("Volume total par direction", pad=20)
    ax1.set_ylim(top=max(total_mb) * 5)   # marge en haut
    for b, v, c in zip(bars1, total_mb, count):
        ax1.text(b.get_x() + b.get_width() / 2, v * 1.3,
                 f"{v:,.0f} Mo\n({c} transferts)",
                 ha="center", va="bottom", fontsize=9)

    # --- 2. Temps total passé par direction ---
    ax2 = axes[1]
    bars2 = ax2.bar(directions, total_ms, color=colors)
    ax2.set_yscale("log")
    ax2.set_ylabel("Temps total (ms, échelle log)")
    ax2.set_title("Temps total par direction", pad=20)
    ax2.set_ylim(top=max(total_ms) * 5)   # marge en haut
    for b, v, p in zip(bars2, total_ms, time_percent):
        ax2.text(b.get_x() + b.get_width() / 2, v * 1.3,
                 f"{v:,.1f} ms\n({p}% du temps mémoire)",
                 ha="center", va="bottom", fontsize=9)

    for ax in axes:
        ax.grid(True, which="both", axis="y", alpha=0.3)

    plt.suptitle("Transferts mémoire CPU <-> GPU (Host <-> Device)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    out_path = "memory_transfers.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Graphique enregistré : {out_path}")

    # --- Résumé texte ---
    print("\n=== Résumé ===")
    for r_size, r_time in zip(size_rows, time_rows):
        print(f"{r_size['direction']:15s} : "
              f"{float(r_size['total_MB']):>10,.1f} Mo total, "
              f"{int(r_size['count']):>3} transferts, "
              f"{float(r_time['total_time_ns'])/1e6:>8,.1f} ms "
              f"({r_time['time_percent']}% du temps mémoire)")


if __name__ == "__main__":
    main()