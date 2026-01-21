import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

def run_experiment_plot(df, x_col, title_prefix, filename):

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    desired_order = ['FlashMLA', 'PyTorch_GQA', 'PyTorch_MHA']
    colors = {'FlashMLA': '#1f77b4', 'PyTorch_GQA': '#2ca02c', 'PyTorch_MHA': '#ff7f0e'}

    ax = ax1
    pivot_time = df.pivot(index=x_col, columns='Method', values='Time_ms')
    existing_cols = [c for c in desired_order if c in pivot_time.columns]
    pivot_time = pivot_time[existing_cols]
    
    color_list = [colors.get(col, '#333333') for col in pivot_time.columns]
    pivot_time.plot(kind='bar', width=0.8, color=color_list, edgecolor='black', linewidth=0.5, ax=ax)
    ax.set_yscale('log')
    ax.set_title(f"{title_prefix}: Latency (Log Scale, Lower Is Better)", fontsize=14, pad=15)
    ax.set_ylabel("Latency (ms)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')

    ax = ax2
    pivot_kv = df.pivot(index=x_col, columns='Method', values='KV_Size_MB')
    valid_cols_kv = [c for c in desired_order if c in pivot_kv.columns]
    pivot_kv = pivot_kv[valid_cols_kv]
    color_list_kv = [colors.get(col, '#333333') for col in pivot_kv.columns]

    pivot_kv.plot(kind='bar', width=0.8, color=color_list_kv, edgecolor='black', linewidth=0.5, ax=ax, legend=False)
    ax.set_yscale('log')
    ax.set_title(f"{title_prefix}: KV Cache Size (Log Scale, Lower Is Better)", fontsize=14, pad=15)
    ax.set_ylabel("Size (MB)", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.4, which='both')

    ax = ax3
    if 'PyTorch_MHA' in pivot_time.columns:
        speedup_df = pivot_time.copy()
        for col in speedup_df.columns:
            speedup_df[col] = pivot_time['PyTorch_MHA'] / pivot_time[col]
        
        speedup_df.plot(kind='line', marker='o', linewidth=2, color=color_list, ax=ax)
        ax.set_title(f"{title_prefix}: Speedup (vs MHA)", fontsize=14, pad=15)
        ax.set_ylabel("Speedup Factor (x)", fontsize=12)
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5)
        ax.grid(True, linestyle='--', alpha=0.4)

    for ax_item in [ax1, ax2]:
        y_min, y_max = ax_item.get_ylim()
        for container in ax_item.containers:
            for rect in container:
                height = rect.get_height()
                if pd.isna(height) or height <= 0:
                    ax_item.text(rect.get_x() + rect.get_width()/2, y_min * 1.05, 'OOM', 
                                 ha='center', va='bottom', color='red', fontsize=8, fontweight='bold', rotation=30)
                else:
                    txt = f"{height/1000:.1f}k" if height >= 1000 else f"{height:.1f}"
                    ax_item.text(rect.get_x() + rect.get_width()/2, height * 1.05, txt, 
                                 ha='center', va='bottom', fontsize=6)

    for ax_item in [ax1, ax2, ax3]:
        ax_item.set_xlabel(x_col, fontsize=12)
        ax_item.tick_params(axis='x', rotation=0)

    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()

def main():
    csv_file = 'benchmark_prefill.csv'
    if not os.path.exists(csv_file):
        print(f"Error: {csv_file} not found.")
        return

    df = pd.read_csv(csv_file)
    
    exp1 = df[df['Experiment'] == 'Var_SeqLen']
    if not exp1.empty:
        run_experiment_plot(exp1, x_col='SeqLen', title_prefix='Prefill', filename='prefill_seqlen.png')

    exp2 = df[df['Experiment'] == 'Var_Batch']
    if not exp2.empty:
        run_experiment_plot(exp2, x_col='Batch', title_prefix='Prefill', filename='prefill_batch.png')

if __name__ == "__main__":
    main()