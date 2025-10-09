import joblib
import pandas as pd
import os
import glob

folder = "./results/results_giammy/experiment_before/run1/logs"   # change if needed
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))

rows = []

for pkl_file in pkl_files:
    print(f"[INFO] Processing {pkl_file} ...")
    data = joblib.load(pkl_file)

    row = {"source_file": os.path.basename(pkl_file)}

    if isinstance(data, dict):
        # If dict, try to store summary values
        for k, v in data.items():
            if isinstance(v, (list, tuple)):
                # Example: store final value + maybe average
                row[f"{k}_last"] = v[-1]
                row[f"{k}_mean"] = sum(v) / len(v)
            else:
                row[k] = v
    elif isinstance(data, pd.DataFrame):
        # Store last row as summary
        for col in data.columns:
            row[f"{col}_last"] = data[col].iloc[-1]
            row[f"{col}_mean"] = data[col].mean()
    else:
        row["value"] = str(data)

    rows.append(row)

# Make one DataFrame with all runs
final_df = pd.DataFrame(rows)

# Save to CSV
output_file = os.path.join(folder, "experiment_summary.csv")
final_df.to_csv(output_file, index=False)

print(f"\n✅ Saved clean summary table to: {output_file}")
print(final_df.head())
