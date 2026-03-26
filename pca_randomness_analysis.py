import argparse
import csv
import importlib
import json
import math
import os
import re
from glob import glob
from collections import Counter
from statistics import mean, pstdev

from parser.parser import Parser
from scheduler.beam_search_scheduler import BeamSearchScheduler


def vector_sub(a, b):
    return [x - y for x, y in zip(a, b)]


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mat_vec_mul(matrix, vec):
    return [dot(row, vec) for row in matrix]


def normalize(vec):
    norm = math.sqrt(dot(vec, vec))
    if norm == 0.0:
        return [0.0 for _ in vec], 0.0
    return [x / norm for x in vec], norm


def power_iteration(cov, iterations=300, tol=1e-10):
    n = len(cov)
    v = [1.0 / math.sqrt(n)] * n
    eig_old = 0.0

    for _ in range(iterations):
        w = mat_vec_mul(cov, v)
        v_new, _ = normalize(w)
        eig = dot(v_new, mat_vec_mul(cov, v_new))
        if abs(eig - eig_old) < tol:
            return eig, v_new
        v = v_new
        eig_old = eig

    eig = dot(v, mat_vec_mul(cov, v))
    return eig, v


def deflate(cov, eig, vec):
    n = len(cov)
    out = [[cov[i][j] - eig * vec[i] * vec[j] for j in range(n)] for i in range(n)]
    return out


def compute_pca_2d(features, standardize=True):
    if len(features) < 2:
        raise ValueError("Need at least 2 samples for PCA")

    dim = len(features[0])
    n = len(features)

    means = [sum(row[d] for row in features) / n for d in range(dim)]
    centered = [[row[d] - means[d] for d in range(dim)] for row in features]

    if standardize:
        stds = []
        denom = max(1, n - 1)
        for d in range(dim):
            var_d = sum((row[d] - means[d]) ** 2 for row in features) / denom
            stds.append(math.sqrt(var_d) if var_d > 0.0 else 1.0)
        centered = [[centered[i][d] / stds[d] for d in range(dim)] for i in range(n)]

    cov = [[0.0] * dim for _ in range(dim)]
    denom = max(1, n - 1)
    for i in range(dim):
        for j in range(dim):
            cov[i][j] = sum(centered[k][i] * centered[k][j] for k in range(n)) / denom

    eig1, vec1 = power_iteration(cov)
    cov2 = deflate(cov, eig1, vec1)
    eig2, vec2 = power_iteration(cov2)

    coords = []
    for row in centered:
        pc1 = dot(row, vec1)
        pc2 = dot(row, vec2)
        coords.append((pc1, pc2))

    total_var = sum(max(cov[i][i], 0.0) for i in range(dim))
    if total_var > 0.0:
        evr1 = max(eig1, 0.0) / total_var
        evr2 = max(eig2, 0.0) / total_var
    else:
        evr1 = 0.0
        evr2 = 0.0

    return coords, (evr1, evr2), {
        "means": means,
        "pc1_vector": vec1,
        "pc2_vector": vec2,
        "standardized": standardize,
    }


def save_loadings(loadings, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fieldnames = ["feature", "pc1_loading", "pc2_loading", "abs_pc1", "abs_pc2"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in loadings:
            writer.writerow(row)


def summarize_solution(sol, opening, closing):
    programs = sol.scheduled_programs
    durations = [s.end - s.start for s in programs]

    total_watched = sum(durations)
    horizon = max(1, closing - opening)
    coverage_ratio = total_watched / horizon

    unique_channels = len({s.channel_id for s in programs})
    switches = 0
    for i in range(1, len(programs)):
        if programs[i].channel_id != programs[i - 1].channel_id:
            switches += 1

    avg_segment = mean(durations) if durations else 0.0
    duration_std = pstdev(durations) if len(durations) >= 2 else 0.0

    channel_counts = Counter(s.channel_id for s in programs)
    if channel_counts and len(programs) > 0:
        dominance_ratio = max(channel_counts.values()) / len(programs)
    else:
        dominance_ratio = 0.0

    features = [
        float(sol.total_score),
        float(len(programs)),
        float(total_watched),
        float(coverage_ratio),
        float(unique_channels),
        float(switches),
        float(avg_segment),
        float(duration_std),
        float(dominance_ratio),
    ]

    return {
        "score": sol.total_score,
        "program_count": len(programs),
        "total_watched": total_watched,
        "coverage_ratio": coverage_ratio,
        "unique_channels": unique_channels,
        "switches": switches,
        "avg_segment": avg_segment,
        "duration_std": duration_std,
        "dominance_ratio": dominance_ratio,
        "features": features,
    }


def summarize_schedule_rows(schedule_rows, score):
    durations = [int(s["end"]) - int(s["start"]) for s in schedule_rows]

    if schedule_rows:
        opening = min(int(s["start"]) for s in schedule_rows)
        closing = max(int(s["end"]) for s in schedule_rows)
    else:
        opening = 0
        closing = 1

    total_watched = sum(durations)
    horizon = max(1, closing - opening)
    coverage_ratio = total_watched / horizon

    unique_channels = len({int(s["channel_id"]) for s in schedule_rows})
    switches = 0
    for i in range(1, len(schedule_rows)):
        if int(schedule_rows[i]["channel_id"]) != int(schedule_rows[i - 1]["channel_id"]):
            switches += 1

    avg_segment = mean(durations) if durations else 0.0
    duration_std = pstdev(durations) if len(durations) >= 2 else 0.0

    channel_counts = Counter(int(s["channel_id"]) for s in schedule_rows)
    if channel_counts and len(schedule_rows) > 0:
        dominance_ratio = max(channel_counts.values()) / len(schedule_rows)
    else:
        dominance_ratio = 0.0

    features = [
        float(score),
        float(len(schedule_rows)),
        float(total_watched),
        float(coverage_ratio),
        float(unique_channels),
        float(switches),
        float(avg_segment),
        float(duration_std),
        float(dominance_ratio),
    ]

    return {
        "score": score,
        "program_count": len(schedule_rows),
        "total_watched": total_watched,
        "coverage_ratio": coverage_ratio,
        "unique_channels": unique_channels,
        "switches": switches,
        "avg_segment": avg_segment,
        "duration_std": duration_std,
        "dominance_ratio": dominance_ratio,
        "features": features,
    }


def parse_score_from_output_filename(path):
    m = re.search(r"_(\d+)\.json$", os.path.basename(path))
    if m:
        return int(m.group(1))
    return 0


def run_from_output_files(file_paths, label):
    rows = []
    for i, p in enumerate(sorted(file_paths), start=1):
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        schedules = data.get("scheduled_programs", [])
        score = parse_score_from_output_filename(p)
        summary = summarize_schedule_rows(schedules, score)
        rows.append({
            "instance": label,
            "run": i,
            "seed": "",
            **summary,
        })
    return rows


def run_instance(
    input_path,
    runs,
    base_seed,
    beam_width,
    lookahead,
    percentile,
    noise_strength,
    n_runs,
):
    instance = Parser(input_path).parse()
    rows = []

    print(f"Running {runs} stochastic scheduler iterations on {os.path.basename(input_path)}...")
    
    for run_idx in range(runs):
        seed = base_seed + run_idx
        scheduler = BeamSearchScheduler(
            instance_data=instance,
            beam_width=beam_width,
            lookahead_limit=lookahead,
            density_percentile=percentile,
            verbose=False,
            noise_strength=noise_strength,
            n_runs=n_runs,
            time_limit=300,
        )
        sol = scheduler.generate_solution()

        summary = summarize_solution(sol, instance.opening_time, instance.closing_time)
        rows.append({
            "instance": os.path.basename(input_path),
            "run": run_idx + 1,
            "seed": seed,
            **summary,
        })
        
        print(f"  ✓ Run {run_idx + 1}/{runs} completed (score={summary['score']})")

    return rows


def save_csv(rows, out_csv):
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)

    fieldnames = [
        "instance",
        "run",
        "seed",
        "score",
        "program_count",
        "total_watched",
        "coverage_ratio",
        "unique_channels",
        "switches",
        "avg_segment",
        "duration_std",
        "dominance_ratio",
        "pc1",
        "pc2",
    ]

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r[k] for k in fieldnames})


def _axis_label(loadings, pc_key, title):
    abs_key = "abs_pc1" if pc_key == "pc1_loading" else "abs_pc2"
    top = sorted(loadings, key=lambda x: x[abs_key], reverse=True)[:3]
    parts = [f"{x['feature']} ({x[pc_key]:+.2f})" for x in top]
    return f"{title} [{', '.join(parts)}]"


def maybe_plot(rows, out_png, loadings):
    try:
        plt = importlib.import_module("matplotlib.pyplot")
    except Exception:
        print("matplotlib not available; skipping PNG plot")
        return

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    groups = {}
    for r in rows:
        groups.setdefault(r["instance"], []).append(r)

    plt.figure(figsize=(10, 7))
    colors = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]

    for idx, (name, points) in enumerate(groups.items()):
        x = [p["pc1"] for p in points]
        y = [p["pc2"] for p in points]
        plt.scatter(x, y, s=80, alpha=0.85, color=colors[idx % len(colors)], label=name)

        # Add run number labels to each point
        for i, p in enumerate(points):
            plt.text(p["pc1"], p["pc2"], f"  R{p['run']}", fontsize=8, alpha=0.7)

        cx = sum(x) / max(1, len(x))
        cy = sum(y) / max(1, len(y))
        plt.scatter([cx], [cy], s=180, marker="X", color=colors[idx % len(colors)], edgecolor="black")

    plt.title("Stochastic Beam Search Distribution (PCA 2D)")
    plt.xlabel(_axis_label(loadings, "pc1_loading", "PC1 / X"))
    plt.ylabel(_axis_label(loadings, "pc2_loading", "PC2 / Y"))
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()


def main():
    ap = argparse.ArgumentParser(description="Run stochastic scheduler and analyze run distribution with PCA")
    ap.add_argument("--mode", choices=["rerun", "outputs"], default="rerun",
                    help="rerun: execute scheduler runs; outputs: read existing JSON output files")
    ap.add_argument("--instance-a", default="data/input/kosovo_tv_input.json", help="First input instance path")
    ap.add_argument("--instance-b", default=None, help="Second input instance path (optional, only if specified)")
    ap.add_argument("--runs", type=int, default=10, help="Number of runs per instance")
    ap.add_argument("--seed", type=int, default=42, help="Base random seed")
    ap.add_argument("--beam-width", type=int, default=100)
    ap.add_argument("--lookahead", type=int, default=4)
    ap.add_argument("--density-percentile", type=int, default=25)
    ap.add_argument("--temperature", type=float, default=0.08, help="Noise strength for stochastic behavior")
    ap.add_argument("--n-runs", type=int, default=3, help="Number of attempts per run; best is kept")
    ap.add_argument("--out-csv", default="data/output/pca_randomness_runs.csv")
    ap.add_argument("--out-png", default="data/output/pca_randomness_scatter.png")
    ap.add_argument("--out-loadings", default="data/output/pca_loadings.csv",
                    help="CSV path for PCA feature loadings (PC1/PC2 weights)")
    ap.add_argument("--raw-pca", action="store_true", help="Disable feature standardization before PCA")
    ap.add_argument("--output-a-glob", default="", help="Glob for existing output JSONs of group A")
    ap.add_argument("--output-b-glob", default="", help="Glob for existing output JSONs of group B (optional)")
    ap.add_argument("--label-a", default="instance_a", help="Label for group A in outputs mode")
    ap.add_argument("--label-b", default="instance_b", help="Label for group B in outputs mode")
    args = ap.parse_args()

    # Show what mode and parameters are being used
    print(f"\n{'='*70}")
    print(f"PCA RANDOMNESS ANALYSIS")
    print(f"{'='*70}")
    print(f"Mode: {args.mode}")
    if args.mode == "rerun":
        print(f"Instance A: {os.path.basename(args.instance_a)} (runs={args.runs})")
        if args.instance_b:
            print(f"Instance B: {os.path.basename(args.instance_b)} (runs={args.runs})")
        print(f"Parameters: noise_strength={args.temperature}, n_runs={args.n_runs}")
    else:
        print(f"Outputs mode: reading from glob patterns")
    print(f"{'='*70}\n")

    if args.mode == "rerun":
        rows_a = run_instance(
            input_path=args.instance_a,
            runs=args.runs,
            base_seed=args.seed,
            beam_width=args.beam_width,
            lookahead=args.lookahead,
            percentile=args.density_percentile,
            noise_strength=args.temperature,
            n_runs=args.n_runs,
        )

        rows_b = []
        if args.instance_b:
            rows_b = run_instance(
                input_path=args.instance_b,
                runs=args.runs,
                base_seed=args.seed + 10_000,
                beam_width=args.beam_width,
                lookahead=args.lookahead,
                percentile=args.density_percentile,
                noise_strength=args.temperature,
                n_runs=args.n_runs,
            )
    else:
        if not args.output_a_glob:
            raise ValueError("In outputs mode you must provide --output-a-glob")

        files_a = glob(args.output_a_glob)
        if len(files_a) < 2:
            raise ValueError("Need at least 2 JSON files in group A for meaningful PCA")

        rows_a = run_from_output_files(files_a, args.label_a)
        rows_b = []
        if args.output_b_glob:
            files_b = glob(args.output_b_glob)
            if len(files_b) < 2:
                raise ValueError("Need at least 2 JSON files in group B when provided")
            rows_b = run_from_output_files(files_b, args.label_b)

    rows = rows_a + rows_b
    if len(rows) < 2:
        raise ValueError("Need at least 2 total samples for PCA")

    feature_names = [
        "score",
        "program_count",
        "total_watched",
        "coverage_ratio",
        "unique_channels",
        "switches",
        "avg_segment",
        "duration_std",
        "dominance_ratio",
    ]

    features = [r["features"] for r in rows]
    coords, (evr1, evr2), pca_info = compute_pca_2d(features, standardize=not args.raw_pca)

    for i, (pc1, pc2) in enumerate(coords):
        rows[i]["pc1"] = pc1
        rows[i]["pc2"] = pc2

    save_csv(rows, args.out_csv)

    loadings = []
    for i, name in enumerate(feature_names):
        pc1 = pca_info["pc1_vector"][i]
        pc2 = pca_info["pc2_vector"][i]
        loadings.append({
            "feature": name,
            "pc1_loading": pc1,
            "pc2_loading": pc2,
            "abs_pc1": abs(pc1),
            "abs_pc2": abs(pc2),
        })
    save_loadings(loadings, args.out_loadings)
    maybe_plot(rows, args.out_png, loadings)

    top_pc1 = sorted(loadings, key=lambda x: x["abs_pc1"], reverse=True)[:3]
    top_pc2 = sorted(loadings, key=lambda x: x["abs_pc2"], reverse=True)[:3]

    print("PCA completed.")
    print(f"Mode: {args.mode}")
    print(f"PCA standardization: {not args.raw_pca}")
    print(f"Explained variance -> PC1: {evr1:.4f}, PC2: {evr2:.4f}")
    print(f"CSV written: {args.out_csv}")
    print(f"PNG written: {args.out_png}")
    print(f"Loadings written: {args.out_loadings}")
    print("Top PC1 contributors:", ", ".join([f"{x['feature']} ({x['pc1_loading']:.3f})" for x in top_pc1]))
    print("Top PC2 contributors:", ", ".join([f"{x['feature']} ({x['pc2_loading']:.3f})" for x in top_pc2]))


if __name__ == "__main__":
    main()
