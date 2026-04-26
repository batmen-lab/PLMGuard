import argparse
import os, sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import average_precision_score
from math import pi

from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parser import load_sequence, get_origin_prot_id, parse_score_by_type, transform_dhr_scores
from utils.metric_utils import run_pairewise_seq_identity_score
from utils.plt_utils import get_method_name, get_color_set

import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

group_names = {
    'origin_to_mut_t1': 'S0 vs S1',
    'mut_t1_to_mut_t2': 'S1 vs S2',
    'mut_t2_to_mut_t3': 'S2 vs S3',
    'mut_t3_to_mut_t4': 'S3 vs S4',
    'mut_t4_to_mut_t5': 'S4 vs S5',
} 


def build_urls(dt, mut_type):
    prefix = f'../data/parsed_result/result_{dt}_astral'
    
    param_prefix = {
        "blastp": "_blosum62_Q11R1",
        "diamond": "_blosum62_Q11R1",
    }.get(dt, "")

    url_origin = f'{prefix}_to_astral{param_prefix}_hit1000.txt'
    url_mut = f"{prefix}_{mut_type}_to_astral{param_prefix}_hit1000.txt"
    assert os.path.exists(url_origin) and os.path.exists(url_mut)  
    
    return url_origin, url_mut        

def retrieve_data(url_origin, url_mut, data_type, save_path):
    origin_search_map_homo, _ = parse_score_by_type(url_origin, data_type)
    decoy_search_map_homo, _ = parse_score_by_type(url_mut, data_type)
    
    homoscore_map = {}
    mut_keys = ['mut_t1', 'mut_t2', 'mut_t3', 'mut_t4', 'mut_t5']
    for key in mut_keys: homoscore_map[key] = {}
    homoscore_map['origin'] = {}  
    
    for decoy_prot_id in decoy_search_map_homo:
        assert '_mut' in decoy_prot_id
        origin_prot_id = get_origin_prot_id(decoy_prot_id)
        origin_homo_arr = origin_search_map_homo[origin_prot_id]
        mut_homo_arr = decoy_search_map_homo[decoy_prot_id]
        
        curr_mkey = ''
        for mkey in mut_keys:
            if mkey in decoy_prot_id: curr_mkey = mkey
        assert len(curr_mkey) > 0

        for prot_id, tgt_id, score in mut_homo_arr:
            key = '{}_{}'.format(prot_id, tgt_id)
            homoscore_map[curr_mkey][key] = score
            
        for prot_id, tgt_id, score in origin_homo_arr:
            key = '{}_{}'.format(prot_id, tgt_id)
            homoscore_map['origin'][key] = score

    df = []
    for mkey1, mkey2 in  [('origin', 'mut_t1'), ('mut_t1', 'mut_t2'), ('mut_t2', 'mut_t3'), ('mut_t3', 'mut_t4'), ('mut_t4', 'mut_t5')]:
        for key in homoscore_map[mkey1]:
            if key in homoscore_map[mkey2]:
                df.append({
                    'key': key,
                    'from': mkey1,
                    'to': mkey2,
                    'group': f"{mkey1}_to_{mkey2}",
                    'delta_score': homoscore_map[mkey1][key] - homoscore_map[mkey2][key],
                    'score_from': homoscore_map[mkey1][key],
                    'score_to': homoscore_map[mkey2][key],
                })
    df = pd.DataFrame(df)
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, f"homo_score_mutantWAG_{data_type}.csv"), index=False)
    return df

def plt_radar_plot(data_all, group_type, data_type, save_dir):
    data_dir = os.path.join(save_dir, 'data')
    score_dic = {}
    for dt in data_type:
        for grp in group_type:
            subgroup_df = data_all[(data_all['group'] == grp) & (data_all['data_type'] == dt)]
            
            score = subgroup_df['delta_score'].values
            if dt == 'dhr':
                score = transform_dhr_scores(score)
            label = (score > 0).astype(int)

            auprc_score = average_precision_score(label, np.abs(score))
            score_dic.setdefault(dt, []).append(auprc_score)
            
    # save score_dic
    score_df = pd.DataFrame(score_dic, index=group_type)
    score_df.to_csv(os.path.join(data_dir, 'metrics.csv'))    
                        
    # plot radar plot
    N = len(group_type)
    colors = get_color_set()
    
    values_list = [score_dic[dt] for dt in data_type]
    labels = [get_method_name(i) for i in data_type]
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    
    for i, (values, label, color) in enumerate(zip(values_list, labels, colors)):
        plot_values = values + values[:1]
        
        plot_kwargs = dict(
            linewidth=3,
            label=label,
            color=color,
        )
        if label == "DCTdomain": plot_kwargs["zorder"] = 10
        
        ax.plot(angles, plot_values, **plot_kwargs)
        ax.fill(angles, plot_values, alpha=0.1, color=color)
    
    max_value = max([max(values) for values in values_list])
    min_value = min([min(values) for values in values_list])
    ax.set_xticks(angles[:-1])
    ax.tick_params(axis='x', which='both', pad=20)
    ax.set_xticklabels([group_names[i] for i in group_type], fontsize=12)
    
    y_ticks = np.linspace(0.5, max_value, 6)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f'{tick:.1f}' for tick in y_ticks], fontsize=8, zorder=12)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(min_value, max_value)
    
    plt.legend(loc='upper left', bbox_to_anchor=(1.1, 1.1), fontsize=12)  
    plt.title('Comparison Across Mutation Groups', size=16, pad=20)
    plt.savefig(f"{save_dir}/radar_plot.pdf", dpi=300, bbox_inches='tight')

def plt_joint_density_plot(data_all, data_type, mut_type, save_dir, sample=True, sample_size=10000):
    cur_save_dir = os.path.join(save_dir, 'joint_density_plots')
    os.makedirs(cur_save_dir, exist_ok=True)
    
    stage_names = {
        'origin': 't0',
        'mut_t1': 't1',
        'mut_t2': 't2',
        'mut_t3': 't3',
        'mut_t4': 't4',
        'mut_t5': 't5'
    }
    
    truncated_range_map = {
        'blastp': {
            'origin_to_mut_t1': (0, 800),
            'mut_t1_to_mut_t2': (0, 250),
            'mut_t2_to_mut_t3': (0, 250),
            'mut_t3_to_mut_t4': (0, 150),
            'mut_t4_to_mut_t5': (0, 120)
        },
        'diamond': {
            'origin_to_mut_t1': (0, 1300),
            'mut_t1_to_mut_t2': (0, 350),
            'mut_t2_to_mut_t3': (0, 300),
            'mut_t3_to_mut_t4': (0, 180),
            'mut_t4_to_mut_t5': (0, 180)
        },    
        'mmseq2': {
            'origin_to_mut_t1': (0, 1200),
            'mut_t1_to_mut_t2': (0, 350),
            'mut_t2_to_mut_t3': (0, 300),
            'mut_t3_to_mut_t4': (0, 170),
            'mut_t4_to_mut_t5': (0, 130)
        },   
        'near': {
            'origin_to_mut_t1': (0, 200),
            'mut_t1_to_mut_t2': (0, 40),
            'mut_t2_to_mut_t3': (0, 40),
            'mut_t3_to_mut_t4': (0, 17),
            'mut_t4_to_mut_t5': (0, 15)
        },    
        'dctdomain': {
            'origin_to_mut_t1': (0, 1.3),
            'mut_t1_to_mut_t2': (0, 0.7),
            'mut_t2_to_mut_t3': (0, 0.6),
            'mut_t3_to_mut_t4': (0, 0.4),
            'mut_t4_to_mut_t5': (0, 0.4)
        },
        'dhr': {
            'origin_to_mut_t1': (80, 180),
            'mut_t1_to_mut_t2': (80, 140),
            'mut_t2_to_mut_t3': (80, 140),
            'mut_t3_to_mut_t4': (80, 130),
            'mut_t4_to_mut_t5': (80, 130)
        },
        'plm': {
            'origin_to_mut_t1': (0, 1.3),
            'mut_t1_to_mut_t2': (0, 0.8),
            'mut_t2_to_mut_t3': (0, 0.7),
            'mut_t3_to_mut_t4': (0, 0.6),
            'mut_t4_to_mut_t5': (0, 0.6)
        },
        'tmvec': {
            'origin_to_mut_t1': (0, 1.5),
            'mut_t1_to_mut_t2': (0, 1.0),
            'mut_t2_to_mut_t3': (0, 0.9),
            'mut_t3_to_mut_t4': (0, 0.8),
            'mut_t4_to_mut_t5': (0, 0.8)
        }
    }
    # subfigure for each data type
    for dt in data_type:
        print(f"Plotting scatter plot for data type: {dt}")
        for mut_stage in data_all['group'].unique():
            subgroup_df = data_all[(data_all['data_type'] == dt) & (data_all['group'] == mut_stage)]
            x_label = stage_names[subgroup_df['from'].iloc[0]]
            y_label = stage_names[subgroup_df['to'].iloc[0]]
            
            if sample:
                if len(subgroup_df) > sample_size:
                    subgroup_df = subgroup_df.sample(n=sample_size, random_state=42)
            
            # score = -score + delta_T for dhr
            if dt == 'dhr':
                subgroup_df['score_from'] = transform_dhr_scores(subgroup_df['score_from'].values)
                subgroup_df['score_to'] = transform_dhr_scores(subgroup_df['score_to'].values)
                                    
            x_min = min(subgroup_df['score_from'].min(), subgroup_df['score_to'].min())
            x_max = max(subgroup_df['score_from'].max(), subgroup_df['score_to'].max())
            
            if dt in truncated_range_map:
                x_min, x_max = truncated_range_map[dt].get(mut_stage, (x_min, x_max))
            else: 
                continue

            g = sns.jointplot(
                data=subgroup_df, x="score_from", y="score_to", 
                kind='kde', fill=True,
                cmap='Blues',
                bw_adjust=5)
            
            g.plot_marginals(sns.histplot, kde=True, linewidth=0.1)
            g.ax_marg_x.set_rasterized(True)
            g.ax_marg_y.set_rasterized(True)
            g.ax_joint.plot([x_min, x_max], [x_min, x_max], ls="--", c=".3", alpha=0.5)
            # set x limits and y limits
            g.ax_joint.set_xlim(x_min, x_max)
            g.ax_joint.set_ylim(x_min, x_max)
            
            g.ax_joint.set_xlabel(x_label, fontsize=16)
            g.ax_joint.set_ylabel(y_label, fontsize=16)

            g.ax_joint.tick_params(axis='x', labelsize=12)
            g.ax_joint.tick_params(axis='y', labelsize=12)
            
            plt.suptitle(f'Homology Score Scatter Plot ({get_method_name(dt)})', fontsize=18)
            plt.tight_layout()
            fname = f"joint_density{'_s' if sample else ''}_{dt}_{mut_stage}_{mut_type}.pdf"
            fig_url = os.path.join(cur_save_dir, fname)
            os.makedirs(os.path.dirname(fig_url), exist_ok=True)
            plt.savefig(fig_url, dpi=300)
            plt.close()  

def plt_len_stat(url, save_dir=None):
    astral_dic = load_sequence(url)
    lengths = [len(seq) for seq in astral_dic.values()]
    
    plt.figure(figsize=(5, 5))
    sns.histplot(
        lengths, bins=50, 
        kde=False, color='#4C91C0', 
        stat='count', edgecolor='none',
        alpha=0.8
        )
    plt.xlabel("Sequence Length", fontsize=12)
    plt.ylabel("Count", fontsize=12)
    plt.title("Distribution of Sequence Lengths in ASTRAL Database", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/astral_sequence_length_distribution.pdf", dpi=300, bbox_inches='tight')
    plt.close()

def plt_mut_rate(url_origin, url_mut, save_dir=None, max_workers=8):
    data_save_url = os.path.join(f"{save_dir}/data", "mutation_rate_data.csv")
    
    if os.path.exists(data_save_url):
        mut_changes_df = pd.read_csv(data_save_url)
        print("Loaded existing mutation rate data.")
    else:
        print("Computing mutation rate data...")
        astral_origin = load_sequence(url_origin)
        astral_mut = load_sequence(url_mut)
        mut_stages = ['mut_t1', 'mut_t2', 'mut_t3', 'mut_t4', 'mut_t5']
        
        tasks = []
        for seq_id in astral_origin:
            seq_origin = astral_origin[seq_id]
            for stage in mut_stages:
                seq_id_stage = f"{seq_id}_{stage}"
                seq_mut = astral_mut.get(seq_id_stage, None)
                if seq_mut is not None:
                    tasks.append((seq_origin, seq_mut, stage))
        
        mut_changes = []
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(process_task, task): task for task in tasks}
            
            for future in tqdm(as_completed(future_to_args), total=len(future_to_args), desc="Processing sequence identity tasks"):
                try:
                    result = future.result()
                    mut_changes.append(result)
                except Exception as e:
                    print(f"Task failed: {e}")
        
        mut_changes_df = pd.DataFrame(mut_changes)
        mut_changes_df.to_csv(data_save_url, index=False)    
    
    # boxplot for each mutation stage
    palette = [plt.cm.Blues(0.3 + i*0.15) for i in range(5)]
    plt.figure(figsize=(5, 5))
    sns.boxplot(
        x='Mutation Stage', y='Sequence Identity', 
        data=mut_changes_df, 
        order=mut_stages,
        showfliers=False,
        width=0.4,
        palette=palette
        )
    plt.xticks(ticks=range(5), labels=['t1', 't2', 't3', 't4', 't5'])
    plt.xlabel("Mutation Stage", fontsize=12)
    plt.ylabel("Sequence Identity", fontsize=12)
    plt.title("Distribution of Sequence Identity Across Mutation Stages (Compared to Original Sequence)", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"{save_dir}/mutation_change_distribution_boxplot.pdf", dpi=300, bbox_inches='tight')
    plt.close() 

def process_task(args):
    seq_origin, seq_mut, stage = args
    seq_id_score, _ = run_pairewise_seq_identity_score(seq_origin, seq_mut)
    return {'Mutation Stage': stage, 'Sequence Identity': seq_id_score}
    
            
def main(args):
    data_type=['blastp', 'diamond', 'mmseq2', 'near', 'dctdomain', 'dhr', 'plm', 'tmvec']
    group_type=['origin_to_mut_t1', 'mut_t1_to_mut_t2', 'mut_t2_to_mut_t3', 'mut_t3_to_mut_t4', 'mut_t4_to_mut_t5']
    save_dir = "../results/check_bio_evolution"
    os.makedirs(save_dir, exist_ok=True)
    
    data_dfs = []
    for dt in data_type:
        url_origin, url_mut = build_urls(dt, args.mut_type)
        data_df = retrieve_data(url_origin, url_mut, dt, f"{save_dir}/data")
        data_df['data_type']=dt
        data_dfs.append(data_df)
    
    data_all = pd.concat(data_dfs, ignore_index=True)
    data_all.to_csv(os.path.join(f"{save_dir}/data", f"homo_score_{args.mut_type}_all.csv"))

    ## summary statistic
    plt_len_stat(url="../data/db/astral.fa", save_path=save_dir)
    plt_mut_rate(url_origin="../data/db/astral.fa", url_mut=f"../data/db/astral_{args.mut_type}.fa", save_path=save_dir)    

    ## quantitative plot
    plt_radar_plot(data_all, group_type, data_type, save_dir)

    ## qualitative analysis
    plt_joint_density_plot(data_all, data_type, args.mut_type, save_dir, sample=True)
    

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity check: Evolutionary decay test")
    parser.add_argument("--mut_type", default="mutantWAG", type=str, help="Decoy type")
    
    args = parser.parse_args()
    main(args)

# command
# python sanity_check/check_bio_evolution.py
