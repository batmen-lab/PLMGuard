import argparse, os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import numpy as np

import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parser import parse_multilvl_homo_result_by_type, transform_dhr_scores
from utils.plt_utils import get_method_name, get_color_set

import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

decoy_type_map = {
    "doubself": "doub_shuf0",
    "doubshuf": "doub_shuf1"
}

def build_urls(dt, decoy_type):
    prefix = f"../data/parsed_result/result_{dt}_astral"
    param_prefix = {
            "blastp": "_blosum62_Q11R1",
            "diamond": "_blosum62_Q11R1",
        }.get(dt, "")

    url_origin = f"{prefix}_to_astral{param_prefix}_hit1000.txt"
    url_decoy = f"{prefix}_{decoy_type}_to_astral{param_prefix}_hit1000.txt" 
    assert os.path.exists(url_origin), f"Origin file not found: {url_origin}"
    assert os.path.exists(url_decoy), f"Decoy file not found: {url_decoy}"  
    
    return url_origin, url_decoy

def retrieve_data(url_origin, url_decoy, data_type, decoy_type, save_dir):
    cur_save_dir = os.path.join(save_dir, 'data')
    os.makedirs(cur_save_dir, exist_ok=True)
    origin_search_map = parse_multilvl_homo_result_by_type(url_origin, data_type)
    decoy_search_map = parse_multilvl_homo_result_by_type(url_decoy, data_type)
    
    origin_search_map_homo = {}
    decoy_search_map_homo = {}
    for key in [1, 2]:
        for qid, tid_dict in origin_search_map[key].items():
            origin_search_map_homo.setdefault(qid, {}).update(tid_dict)

        for qid, tid_dict in decoy_search_map[key].items():
            decoy_search_map_homo.setdefault(qid, {}).update(tid_dict)

    data_df = []
    for origin_qid in origin_search_map_homo:
        decoy_qid = f"{origin_qid}_{decoy_type_map[decoy_type]}"
        
        for origin_tid in origin_search_map_homo[origin_qid]:
            origin_score, _ = origin_search_map_homo[origin_qid][origin_tid]
            
            if decoy_qid in decoy_search_map_homo and origin_tid in decoy_search_map_homo[decoy_qid]:
                decoy_score, _ = decoy_search_map_homo[decoy_qid][origin_tid]
            else:
                continue
            
            data_df.append({
                'qid': origin_qid,
                'tid': origin_tid,
                'origin_score': origin_score,
                'decoy_score': decoy_score
            })
    data_df = pd.DataFrame(data_df)
    data_df.to_csv(os.path.join(cur_save_dir, f"homo_score_{decoy_type}_{data_type}.csv"))
    return data_df   

def plt_joint_density_plots(data_all, data_type, decoy_type, save_dir, sample=True, sample_size=10000):
    cur_save_dir = os.path.join(save_dir, f'joint_density_plots/{decoy_type}')
    truncated_range_map = {
        'blastp': {
            'doubshuf': (0, 400),
            'doubself': (0, 400),
        },
        'diamond': {
            'doubshuf': (0, 500),  
            'doubself': (0, 500), 
        },
        'mmseq2': {
            'doubshuf': (0, 500),
            'doubself': (0, 500),
            },   
        'near': {
            'doubshuf': (0, 150),  
            'doubself': (0, 200),
        },  
        'tmvec': {
            'doubshuf': (0, 1.3),
            'doubself': (0, 1.3)}, 
        'plm': {
            'doubshuf': (0, 1.1),
            'doubself': (0, 1.1)
            },  
        'dctdomain': {
            'doubshuf': (0, 1.1),
            'doubself': (0, 1.1)
            },  
        'dhr': {
            'doubshuf': (80, 170),
            'doubself': (80, 170)
            },
    }
    
    # subfigure for each data type
    for dt in data_type:
        print(f"Plotting scatter plot for data type: {dt}")
        subgroup_df = data_all[data_all['data_type'] == dt]
        
        # for better visualization, we subsample 1k data
        if sample:
            if len(subgroup_df) > sample_size:
                subgroup_df = subgroup_df.sample(n=sample_size, random_state=42)
        
        # score = -score + delta_T for dhr
        if dt == 'dhr':
            subgroup_df['origin_score'] = transform_dhr_scores(subgroup_df['origin_score'].values)
            subgroup_df['decoy_score'] = transform_dhr_scores(subgroup_df['decoy_score'].values)
                                
        x_min = min(subgroup_df['origin_score'].min(), subgroup_df['decoy_score'].min())
        x_max = max(subgroup_df['origin_score'].max(), subgroup_df['decoy_score'].max())
        
        x_min, x_max = truncated_range_map[dt].get(decoy_type, (x_min, x_max))
        g = sns.jointplot(
            data=subgroup_df, x="origin_score", y="decoy_score", 
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
        
        g.ax_joint.set_xlabel('Original Homology Score', fontsize=16)
        g.ax_joint.set_ylabel('Double-length Homology Score', fontsize=16)

        g.ax_joint.tick_params(axis='x', labelsize=12)
        g.ax_joint.tick_params(axis='y', labelsize=12)
        
        plt.suptitle(f'Homology Score Scatter Plot ({get_method_name(dt)})', fontsize=18)
        plt.tight_layout()
        fname = f"joint_density{'_s' if sample else ''}_{decoy_type}_{dt}.pdf"
        fig_url = os.path.join(cur_save_dir, fname)
        os.makedirs(os.path.dirname(fig_url), exist_ok=True)
        plt.savefig(fig_url, dpi=300)
        plt.close() 
            
def compare_metric(data_type, data_dir):
    results_df_url = os.path.join(data_dir, f"data/metrics.csv")
    if os.path.exists(results_df_url):
        print(f"Metrics exist at {results_df_url}, loading...")
        results_df = pd.read_csv(results_df_url)
    else:
        results_df = []
        
        for decoy_type in ['doubshuf', 'doubself']:
            for dt in data_type:
                data_url = os.path.join(data_dir, f"data/homo_score_{decoy_type}_{dt}.csv")
                data_df = pd.read_csv(data_url)
                
                x_quant = data_df['origin_score'].quantile(0.99)
                X = data_df['origin_score'].values
                y = data_df['decoy_score'].values       
                
                # distance to y=x
                distances = np.abs(y - X) / np.sqrt(2)
                avg_d = np.mean(distances / x_quant)

                results_df.append({
                    'decoy_type': decoy_type, 
                    'data_type': dt, 
                    'avg_norm_dist': avg_d})

        results_df = pd.DataFrame(results_df)
        results_df.to_csv(results_df_url, index=False)
        
    # lollipop plot for avg normalized distance
    doubself_values = []
    doubshuf_values = []
    for dt in data_type:
        metric = results_df[(results_df['data_type']==dt) & (results_df['decoy_type']=='doubself')]['avg_norm_dist'].values[0]
        doubself_values.append(metric)
        metric = results_df[(results_df['data_type']==dt) & (results_df['decoy_type']=='doubshuf')]['avg_norm_dist'].values[0]
        doubshuf_values.append(metric)
        
    fig, ax = plt.subplots(figsize=(10,4))
    palette = get_color_set()  
      
    y_line = 0
    offset = 0.15
    x = np.arange(len(doubself_values))

    for i, (xi, yi) in enumerate(zip(x, doubself_values)):
        label = 'Double self' if i==0 else None
        plt.plot([xi-offset, xi-offset], [y_line, yi], color=palette[i], alpha=0.7)
        plt.scatter(xi-offset, yi, color=palette[i], s=100, marker='o', label=label)

    for i, (xi, yi) in enumerate(zip(x, doubshuf_values)):
        label = 'Double shuffle' if i==0 else None
        plt.plot([xi+offset, xi+offset], [y_line, yi], color=palette[i], alpha=0.7)
        plt.scatter(xi+offset, yi, color=palette[i], s=100, marker='s', label=label)                
        
    xmin, xmax = plt.xlim()
    plt.hlines(y=y_line, xmin=xmin, xmax=xmax, linewidth=1, colors='gray', linestyles='--', alpha=0.7)
    
    ax.set_xticks(np.arange(len(data_type)))         
    ax.set_xticklabels([get_method_name(dt) for dt in data_type], rotation=45)
    ax.set_ylabel('Avg Normalized Distance')
    ax.set_title("Avg Normalized Distance of Linear Regression between Origin and Decoy Scores")
    
    plt.subplots_adjust(bottom=0.3) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    fig_url = os.path.join(data_dir, f"metric_lollipop_avg_norm_dist.pdf")
    plt.savefig(fig_url, dpi=300)   
    plt.close()

     
def main():
    data_type=['blastp', 'diamond', 'mmseq2', 'near', 'dctdomain', 'dhr', 'plm', 'tmvec']
    save_dir = "../results/check_pert_doublen"
    os.makedirs(save_dir, exist_ok=True)
    
    for decoy_type in ['doubshuf', 'doubself']:
        print(f"Processing decoy type: {decoy_type}")
        data_dfs = []
        for dt in data_type:
            url_origin, url_decoy = build_urls(dt, decoy_type)
            data_df = retrieve_data(url_origin, url_decoy, dt, decoy_type, save_dir)
            data_df['data_type']=dt
            data_dfs.append(data_df)
        
        data_all = pd.concat(data_dfs, ignore_index=True)
        
        ## qualitative analysis
        plt_joint_density_plots(data_all, data_type, decoy_type, save_dir, sample=True)
        
    ## Metric
    compare_metric(data_type, save_dir)
    
    
if __name__ == "__main__":
    main()

# command
# python sanity_check/check_pert_doublen.py