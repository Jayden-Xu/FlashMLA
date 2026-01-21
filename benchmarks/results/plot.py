
import pandas as pd
import matplotlib.pyplot as plt
import os

def run_experiment_plot(df, x_col, title_prefix, filename):

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
    metrics = [
        {'col': 'Time_ms', 'title': 'Latency (Lower is Better)', 'ylabel': 'Latency (ms)', 'ax': ax1},
        {'col': 'KV_Size_MB', 'title': 'KV Cache Size (Lower is Better)', 'ylabel': 'Size (MB)', 'ax': ax2}
    ]
    
    desired_order = ['FlashMLA', 'PyTorch_GQA', 'PyTorch_MHA']
    colors = {'FlashMLA': '#1f77b4', 'PyTorch_GQA': '#2ca02c', 'PyTorch_MHA': '#ff7f0e'}

    for m in metrics:
        ax = m['ax']
        pivot_df = df.pivot(index=x_col, columns='Method', values=m['col'])
        existing_cols = [c for c in desired_order if c in pivot_df.columns]
        pivot_df = pivot_df[existing_cols]
        
        color_list = [colors.get(col, '#333333') for col in pivot_df.columns]

        pivot_df.plot(kind='bar', width=0.8, color=color_list, edgecolor='black', 
                      linewidth=0.5, ax=ax, legend=True if m['col'] == 'Time_ms' else False)
        
        ax.set_title(f"{title_prefix}: {m['title']}", fontsize=16, pad=15)
        ax.set_xlabel(x_col, fontsize=12)
        ax.set_ylabel(m['ylabel'], fontsize=12)
        ax.grid(axis='y', linestyle='--', alpha=0.4)
        ax.tick_params(axis='x', rotation=0)

        y_bottom, y_top = ax.get_ylim()
        for container in ax.containers:
            label = str(container.get_label())
            for i, rect in enumerate(container):
                height = rect.get_height()
                x = rect.get_x() + rect.get_width() / 2
                
                idx_val = pivot_df.index[i]
                val = pivot_df.loc[idx_val, label]
                
                if pd.isna(val) or val == 0:
                    ax.text(x, y_bottom, 'OOM', ha='center', va='bottom', 
                             color='red', fontsize=10, fontweight='bold')
                else:
                    txt = f"{val/1000:.1f}k" if val >= 1000 else f"{val:.0f}"
                    ax.text(x, height + (y_top * 0.01), txt, ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    print(f"Saving {filename}...")
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
        run_experiment_plot(
            exp1, 'Var_SeqLen', x_col='SeqLen', 
            title_prefix='Prefill',
            filename='exp1_combined_seqlen.png'
        )

    exp2 = df[df['Experiment'] == 'Var_Batch']
    if not exp2.empty:
        run_experiment_plot(
            exp2, 'Var_Batch', x_col='Batch', 
            title_prefix='Prefill',
            filename='exp2_combined_batch.png'
        )

if __name__ == "__main__":
    main()