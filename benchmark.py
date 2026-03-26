"""Benchmark: Run each instance 10 times, compare to constructive baselines."""
import sys
import os
import time

from parser.parser import Parser
from scheduler.beam_search_scheduler import BeamSearchScheduler
from utils.utils import Utils

# Instance file -> baseline score mapping
INSTANCES = {
    "toy.json":                  None,   # didactic, no baseline
    "croatia_tv_input.json":     2203,
    "germany_tv_input.json":     1553,
    "kosovo_tv_input.json":      2587,
    "netherlands_tv_input.json": 2636,
    "uk_tv_input.json":          2266,
    "usa_tv_input.json":         3601,
    "australia_iptv.json":       4117,
    "france_iptv.json":          4370,
    "spain_iptv.json":           4555,
    "uk_iptv.json":              5192,
    "us_iptv.json":              4361,
    "singapore_pw.json":         4316,
    "canada_pw.json":            4628,
    "china_pw.json":             2861,
}

N_RUNS = 10
DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "input")

def run_benchmark():
    results = {}
    
    print(f"{'Instance':<28} {'Baseline':>8} | {'Min':>6} {'Avg':>8} {'Max':>6} {'Std':>6} | {'Avg%':>7} {'Max%':>7} | {'Times (s)'}")
    print("-" * 120)
    
    for filename, baseline in INSTANCES.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"{filename:<28} FILE NOT FOUND")
            continue
        
        # Parse instance
        parser = Parser(filepath)
        instance = parser.parse()
        Utils.set_current_instance(instance)
        
        n_ch = len(instance.channels)
        n_prog = sum(len(ch.programs) for ch in instance.channels)
        
        scores = []
        times_list = []
        
        for i in range(N_RUNS):
            t0 = time.time()
            scheduler = BeamSearchScheduler(
                instance_data=instance,
                beam_width=100,
                lookahead_limit=4,
                density_percentile=25,
                verbose=False,
                noise_strength=0.08,
                n_runs=3,
                time_limit=300
            )
            sol = scheduler.generate_solution()
            elapsed = time.time() - t0
            scores.append(sol.total_score)
            times_list.append(elapsed)
            # Progress dot
            sys.stdout.write(f"\r  {filename}: run {i+1}/{N_RUNS} = {sol.total_score} ({elapsed:.1f}s)")
            sys.stdout.flush()
        
        sys.stdout.write("\r" + " " * 80 + "\r")  # clear progress line
        
        avg_score = sum(scores) / len(scores)
        min_score = min(scores)
        max_score = max(scores)
        std_score = (sum((s - avg_score)**2 for s in scores) / len(scores)) ** 0.5
        avg_time = sum(times_list) / len(times_list)
        
        if baseline and baseline > 0:
            avg_pct = ((avg_score - baseline) / baseline) * 100
            max_pct = ((max_score - baseline) / baseline) * 100
            avg_pct_str = f"{avg_pct:+.1f}%"
            max_pct_str = f"{max_pct:+.1f}%"
        else:
            avg_pct_str = "N/A"
            max_pct_str = "N/A"
        
        # Show all 10 scores compactly
        scores_str = ", ".join(str(s) for s in scores)
        
        label = filename.replace("_input", "").replace(".json", "")
        print(f"{label:<28} {baseline or 'N/A':>8} | {min_score:>6} {avg_score:>8.1f} {max_score:>6} {std_score:>6.1f} | {avg_pct_str:>7} {max_pct_str:>7} | {avg_time:>5.1f}s  [{scores_str}]")
        
        results[filename] = {
            "baseline": baseline,
            "scores": scores,
            "avg": avg_score,
            "min": min_score,
            "max": max_score,
            "std": std_score,
            "avg_time": avg_time
        }
    
    # Summary
    print("\n" + "=" * 120)
    print("SUMMARY")
    print("=" * 120)
    
    better_avg = 0
    worse_avg = 0
    total_with_baseline = 0
    
    for filename, data in results.items():
        if data["baseline"] and data["baseline"] > 0:
            total_with_baseline += 1
            pct = ((data["avg"] - data["baseline"]) / data["baseline"]) * 100
            if pct >= 0:
                better_avg += 1
            else:
                worse_avg += 1
    
    print(f"Instances with avg >= baseline: {better_avg}/{total_with_baseline}")
    print(f"Instances with avg < baseline:  {worse_avg}/{total_with_baseline}")


if __name__ == "__main__":
    run_benchmark()
