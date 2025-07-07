#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 8 ]; then
  echo "Usage: $0 cfg <num> idea <num> win <num> k <num>"
  exit 1
fi

# Parse arguments
cfg_prefix=$1
cfg_num=$(printf "%03d" "$2")
idea_prefix=$3
idea_num=$4
win_prefix=$5
win_num=$6
k_prefix=$7
k_num=$8

# Construct directories
cfg_dir="${cfg_prefix}_${cfg_num}, ${idea_prefix}:${idea_num}"
sub_dir="${win_prefix}${win_num}_K${k_num}"
base_dir="${cfg_dir}/${sub_dir}"

# Create timestamp to avoid overwrites
timestamp=$(date +%Y%m%d_%H%M%S)

# Prepare output CSV name with timestamp
output_file="evaluate_results_${cfg_prefix}_${cfg_num}_${idea_prefix}${idea_num}_${win_prefix}${win_num}_K${k_num}_${timestamp}.csv"

echo "Processing events in: $base_dir"
echo "Will save results to: $output_file"

# Execute Python for processing
python3 - "$base_dir" "$output_file" << 'PYTHON'
import sys, os, glob, pandas as pd

# Retrieve arguments
base_dir, output_file = sys.argv[1], sys.argv[2]

# Load price table
prices_df = pd.read_excel('60_product_prices_usd.xlsx')
prices = dict(zip(prices_df['Product Name'], prices_df['Price (USD)']))
products = prices_df['Product Name'].tolist()

# Discover event files
pattern = os.path.join(base_dir, 'event_*.txt')
event_files = sorted(glob.glob(pattern))
if not event_files:
    sys.exit(f"No event files found in {base_dir}")

# Initialize previous counts
prev_counts = dict.fromkeys(products, 0)
records = []

for ev_file in event_files:
    ev_name = os.path.splitext(os.path.basename(ev_file))[0]
    df = pd.read_csv(ev_file, sep='\t', header=None, names=['Class','Product','Count'])
    curr_counts = {row['Product']: row['Count'] for _, row in df.iterrows()}
    record = {'event': ev_name}
    total_amount = 0.0
    # Include all products regardless of inventory
    for p in products:
        count = curr_counts.get(p, 0)
        diff = count - prev_counts.get(p, 0)
        # Format diff: +N for increase, -N for decrease, 0 for no change
        if diff > 0:
            diff_str = f"+{diff}"
        elif diff < 0:
            diff_str = str(diff)
        else:
            diff_str = '0'
        record[f"{p}_diff"] = diff_str
        record[f"{p}_inv"] = count
        total_amount += count * prices.get(p, 0)
    record['total_amount'] = round(total_amount, 2)
    records.append(record)
    prev_counts = curr_counts.copy()

# Save results
out_df = pd.DataFrame(records)
out_df.to_csv(output_file, index=False)
print(f"Results saved to {output_file}")
PYTHON
