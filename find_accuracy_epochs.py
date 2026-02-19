import os
import re
import argparse

def find_epochs(log_file):
    if not os.path.exists(log_file):
        print(f"Log file not found: {log_file}")
        return

    print(f"Analyzing log: {log_file}")

    # Store max acc per epoch
    epoch_accs = {}
    current_epoch = -1

    with open(log_file, 'r') as f:
        for line in f:
            # Detect epoch change "epoch: X, flag: evaluation"
            epoch_match = re.search(r'epoch:\s+(\d+),\s+flag:\s+evaluation', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
                continue

            # Detect Metrics line: "20AUC: 0.8123, ACC: 0.7543"
            # We care about the final step (20) usually, or we can track max across steps?
            # The prompt asks "at which epoch". Usually we look at the metric at the end of the test (step 20).
            if "ACC:" in line and "20AUC" in line: # specific to step 20 based on log format?
                 # Log format example: "20AUC: 0.7654, ACC: 0.7123"
                 # Or "Mean Metrics: ... 20AUC: ..., ACC: ..."
                 parts = line.split(',')
                 for p in parts:
                     if "ACC" in p:
                         try:
                             acc_val = float(p.split(':')[1].strip())
                             if current_epoch != -1:
                                 epoch_accs[current_epoch] = acc_val
                         except:
                             pass

    # Thresholds to find
    targets = [0.70, 0.75, 0.80, 0.85]
    found = {t: None for t in targets}

    sorted_epochs = sorted(epoch_accs.keys())
    for epoch in sorted_epochs:
        acc = epoch_accs[epoch]
        print(f"Epoch {epoch}: ACC {acc}")
        for t in targets:
            if found[t] is None and acc >= t:
                found[t] = epoch

    print("\nResults:")
    for t in targets:
        if found[t] is not None:
            print(f"ACC >= {t}: Epoch {found[t]}")
        else:
            print(f"ACC >= {t}: Not reached")

if __name__ == '__main__':
    # Find the latest log file in baseline_log/dbekt22
    log_dir = 'GMOCAT-modif/baseline_log/dbekt22'
    files = [os.path.join(log_dir, f) for f in os.listdir(log_dir) if f.endswith('.txt')]
    if files:
        latest_file = max(files, key=os.path.getctime)
        find_epochs(latest_file)
    else:
        print("No log files found.")
