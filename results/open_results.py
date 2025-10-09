"""
This script processes all .pkl log files in the ./results/logs folder
and generates a summary CSV file.

For each pickle file:
- If it contains a dictionary: store direct values, and for lists/tuples, 
  also compute the last element and the mean.
- If it contains a DataFrame: record the last entry and mean of each column.
- If it contains some other type: save it as a string.

All results are collected into a single Pandas DataFrame and saved as
'experiment_summary.csv' in the same folder, giving a clean overview of 
the experiments.
"""

import joblib
import pandas as pd
import os
import glob

folder = "./results/logs"  
pkl_files = glob.glob(os.path.join(folder, "*.pkl"))

rows = []

for pkl_file in pkl_files:
    print(f"[INFO] Processing {pkl_file} ...")
    data = joblib.load(pkl_file)

    row = {"source_file": os.path.basename(pkl_file)}

    if isinstance(data, dict):

        for k, v in data.items():
            if isinstance(v, (list, tuple)):
               
                row[f"{k}_last"] = v[-1]
                row[f"{k}_mean"] = sum(v) / len(v)
            else:
                row[k] = v
    elif isinstance(data, pd.DataFrame):
    
        for col in data.columns:
            row[f"{col}_last"] = data[col].iloc[-1]
            row[f"{col}_mean"] = data[col].mean()
    else:
        row["value"] = str(data)

    rows.append(row)

final_df = pd.DataFrame(rows)

output_file = os.path.join(folder, "experiment_summary.csv")
final_df.to_csv(output_file, index=False)

print(f"\n Saved clean summary table to: {output_file}")
print(final_df.head())
