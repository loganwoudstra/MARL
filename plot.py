import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy import stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--folders", type=str, nargs="+", default=["data"],
                        help="one or more paths to folders containing CSVs")
    parser.add_argument("--keyword", type=str, default="returns",
                        help="keyword to filter csv files")
    args = parser.parse_args()
    keyword = args.keyword
    folders = args.folders

    # Collect series per folder x config
    all_series = []  # list of dicts: {label, config, running_avg, episode_returns}

    for folder in folders:
        label = os.path.basename(os.path.dirname(os.path.normpath(folder)))
        print(f"Processing folder '{folder}' -> label '{label}'")
        series = load_folder(folder, keyword)
        for s in series:
            algorithm = label.split('_')[0]
            using_bfs = 'bfs' in label
            using_global = 'global' in label
            using_shared = 'shared' in label
            
            
            bfs_string = "+BFS" if using_bfs else ""
            obs_string = 'global obs' if using_global else 'local obs'
            obs_string = obs_string if using_bfs else ""
            shared_string = " (shared reward)" if using_shared else ""
            label = f"{algorithm.upper()}{bfs_string}, {obs_string}{shared_string}"
            s["label"] = label
            all_series.append(s)

    if not all_series:
        print("No data found.")
        return

    # One plot per config, all folders overlaid
    configs = sorted(set(s["config"] for s in all_series))
    for config in configs:
        subset = [s for s in all_series if s["config"] == config]
        plot_config(subset, config, keyword)

def extract_config(filename_without_ext):
    configs = [
        "overcooked_cramped_room_v0",
        "overcooked_forced_coordination_v0",
        "overcooked_coordination_ring_v0",
        "overcooked_counter_circuit_v0",
    ]
    for cfg in configs:
        if cfg in filename_without_ext:
            return cfg
    return None


def load_folder(folder_name, keyword):
    """
    Read all matching CSVs in folder_name, group by config.
    Returns a list of dicts: {config, running_avg, episode_returns}
    """
    configs = [
        "overcooked_cramped_room_v0",
        "overcooked_forced_coordination_v0",
        "overcooked_coordination_ring_v0",
        "overcooked_counter_circuit_v0",
    ]
    data_dict = {cfg: [] for cfg in configs}

    num_envs = 1
    for file in os.listdir(folder_name):
        if file.endswith(".txt"):
            with open(os.path.join(folder_name, file)) as f:
                for line in f:
                    if line.startswith("num_envs:"):
                        num_envs = int(line.split(":")[1].strip())
                        break
                
    for file in os.listdir(folder_name):
        full_path = os.path.join(folder_name, file)
        if (os.path.isfile(full_path)
                and os.path.splitext(file)[-1] == ".csv"
                and keyword in file):
            config = extract_config(os.path.splitext(file)[0])
            assert config is not None, f"{file} is not in the required format."
            print(f"  Reading {full_path}")
            df = pd.read_csv(full_path)
            data_dict[config].append(np.squeeze(df.values))

    results = []
    for cfg in configs:
        if data_dict[cfg]:
            results.append({
                "config": cfg,
                "episode_returns": data_dict[cfg],
                "num_envs": num_envs
            })
    return results

def plot_config(series_list, config, keyword):
    """
    Plot all folders for a single config on one figure.
    Each folder gets its own colour; line = mean, band = 95% CI.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = plt.cm.tab10(range(len(series_list)))
    
    # BFS baseline (only for returns)
    if keyword == "returns":
        bfs = _bfs_baseline(config)
        if bfs is not None:
            ax.axhline(y=bfs, color="black", linestyle="--", linewidth=1.2,
                       label="Single-agent BFS")

    for idx, s in enumerate(series_list):
        color = colors[idx]
        label = s["label"]
        for x in s["episode_returns"]:
            print(label, x.shape)
        print(type(s["episode_returns"][0]))
        
        min_len = min(len(r) for r in s["episode_returns"])
        data_array = np.array([r[:min_len] for r in s["episode_returns"]])
        mean, lo, hi = compute_ci(data_array)
        running_avg = smooth(mean)
        lo = smooth(lo)
        hi = smooth(hi)
        x_coords = [i * s["num_envs"] for i in range(1, len(running_avg) + 1)]
        max_steps = 3_000
        cutoff = next((i for i, x in enumerate(x_coords) if x > max_steps), len(x_coords))
        x_coords, running_avg, lo, hi = x_coords[:cutoff], running_avg[:cutoff], lo[:cutoff], hi[:cutoff]

        ax.fill_between(x_coords, lo, hi, color=color, alpha=0.2)
        ax.plot(x_coords, running_avg, color=color, linewidth=1, label=label)
    
    ax.set_title(config)
    ax.set_xlabel("episode")
    ax.set_ylabel("Return" if keyword == "returns" else keyword)
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    ax.grid(True)
    ax.set_ylim(bottom=0)

    safe_config = config.replace("/", "_")
    out_path = f"plots/{safe_config}_{keyword}.png"
    os.makedirs("plots", exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


def _bfs_baseline(config):
    if "cramped_room" in config:
        return 30.7
    if "counter_circuit" in config:
        return 21.2
    if "coordination_ring" in config:
        return 35.3
    return None

def compute_ci(data_array, confidence=0.95):
    """
    Mean and 95% CI across seeds (t-distribution, robust for small n).
    Args:
        data_array: (num_seeds, num_steps)
    Returns:
        mean, lower_bound, upper_bound  -- each shape (num_steps,)
    """
    n = data_array.shape[0]
    mean = np.mean(data_array, axis=0)
    if n < 2:
        return mean, mean, mean
    se = stats.sem(data_array, axis=0)
    h = se * stats.t.ppf((1 + confidence) / 2, df=n - 1)
    return mean, mean - h, mean + h


def smooth(arr, window=10):
    """Symmetric moving average with half-window `window`."""
    smoothed = arr.copy().astype(float)
    for i in range(len(arr)):
        smoothed[i] = np.mean(arr[max(0, i - window): min(len(arr), i + window)])
    return smoothed


if __name__ == "__main__":
    main()