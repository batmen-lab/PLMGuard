import argparse, os
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
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
    "trunchalf": "trunc_50pct_rep0",
    "truncqrt": "trunc_25pct_rep0"
}


def plt_compare_stage_joint_density_plot(data_type, save_dir, sample=True, sample_size=10000):
    cur_save_dir = os.path.join(save_dir, "joint_density_plots")
    data_dir = os.path.join(save_dir, "data")
    truncated_range_map = {
        'full_half': {
            'blastp': (0, 800),
            'diamond': (0, 1200),
            'mmseq2': (0, 1200),
            'near': (0, 220),
            'tmvec': (0, 1.3),
            'plm': (0, 1.1),
            'dctdomain': (0, 1.2),
            'dhr': (80, 170),              
            },
        'half_qrt': {
            'blastp': (0, 400),
            'diamond': (0, 600),
            'mmseq2': (0, 600),
            'near': (0, 100),
            'tmvec': (0.2, 1.1),
            'plm': (0.1, 1.1),
            'dctdomain': (0, 0.7),
            'dhr': (70, 170),            
        }
    }
  
    for dt in data_type:
        df_half = pd.read_csv(os.path.join(data_dir, f"homo_score_trunchalf_{dt}.csv"))
        df_qrt  = pd.read_csv(os.path.join(data_dir, f"homo_score_truncqrt_{dt}.csv"))
        
        subgroup_df = pd.merge(
            df_half, df_qrt, on=['qid', 'tid'], 
            suffixes=('_half', '_qrt'),
            ) 
        
        # for better visualization, we subsample 1k data
        if sample and len(subgroup_df) > sample_size:
            subgroup_df = subgroup_df.sample(n=sample_size, random_state=42)

        if dt == 'dhr':
            subgroup_df['origin_score_half'] = transform_dhr_scores(subgroup_df['origin_score_half'])
            subgroup_df['decoy_score_half'] = transform_dhr_scores(subgroup_df['decoy_score_half'])
            subgroup_df['decoy_score_qrt'] = transform_dhr_scores(subgroup_df['decoy_score_qrt'])
        
        for col_name_pair, show_name_pair, type in zip(
            [('origin_score_half', 'decoy_score_half'), ('decoy_score_half', 'decoy_score_qrt')],
            [('Full-length', 'Half-length'), ('Half-length', 'Quater-length')],
            ['full_half', 'half_qrt']
            ):
            col_x, col_y = col_name_pair
            show_x, show_y = show_name_pair
            
            x_min = min(subgroup_df[col_x].min(), subgroup_df[col_y].min())
            x_max = max(subgroup_df[col_x].max(), subgroup_df[col_y].max())
            x_min, x_max = truncated_range_map[type][dt]
            
            g = sns.jointplot(
                data=subgroup_df, x=col_x, y=col_y, 
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
            
            g.ax_joint.set_xlabel(f'{show_x} Homology Score', fontsize=16)
            g.ax_joint.set_ylabel(f'{show_y} Homology Score', fontsize=16)

            g.ax_joint.tick_params(axis='x', labelsize=12)
            g.ax_joint.tick_params(axis='y', labelsize=12)
            
            plt.suptitle(f'Homology Score Scatter Plot ({get_method_name(dt)})', fontsize=18)
            plt.tight_layout()
            fname = f"joint_density{'_s' if sample else ''}_{type}_{dt}.pdf"
            fig_url = os.path.join(f"{cur_save_dir}/{type}", fname)
            os.makedirs(os.path.dirname(fig_url), exist_ok=True)
            plt.savefig(fig_url, dpi=300)
            plt.close()                    
            
def compare_metric(data_type, save_dir):
    data_dir = os.path.join(save_dir, "data")
    results = []
    for dt in data_type:
        df_half = pd.read_csv(os.path.join(data_dir, f"homo_score_trunchalf_{dt}.csv"))
        df_qrt  = pd.read_csv(os.path.join(data_dir, f"homo_score_truncqrt_{dt}.csv"))
        
        subgroup_df = pd.merge(df_half, df_qrt, on=['qid', 'tid'], 
                             suffixes=('_half', '_qrt')) 
        
        if dt == 'dhr':
            subgroup_df['origin_score_half'] = transform_dhr_scores(subgroup_df['origin_score_half'])
            subgroup_df['origin_score_qrt'] = transform_dhr_scores(subgroup_df['origin_score_qrt'])
            subgroup_df['decoy_score_half'] = transform_dhr_scores(subgroup_df['decoy_score_half'])
            subgroup_df['decoy_score_qrt'] = transform_dhr_scores(subgroup_df['decoy_score_qrt'])
        
        y1 = subgroup_df['origin_score_half']
        y2 = subgroup_df['decoy_score_half']
        y3 = subgroup_df['decoy_score_qrt']
        
        x_quant1 = y1.quantile(0.99)
        x_quant2 = y2.quantile(0.99)
        
        subgroup_df['loss1'] = np.maximum(0, y2 - y1) / x_quant1
        subgroup_df['loss2'] = np.maximum(0, y3 - y2) / x_quant2
        subgroup_df['triple_loss'] = (
            subgroup_df['loss1'] + subgroup_df['loss2']
        )
        results.append({
            "data_type": dt,
            "monotonicity_loss": subgroup_df['triple_loss'].mean(),
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(data_dir, f"metrics.csv"), index=False)
    
    # lollipop plot
    fig, ax = plt.subplots(figsize=(10,4))
    palette = get_color_set()
    y_line = 0
    values = []
    for dt in data_type:
        metric = results_df[results_df['data_type']==dt]['monotonicity_loss'].values[0]
        values.append(metric)
    
    x = np.arange(len(values))
    for i, (xi, yi) in enumerate(zip(x, values)):
        plt.plot([xi, xi], [y_line, yi], color=palette[i], alpha=0.7)
        plt.scatter(xi, yi, color=palette[i], s=100, marker='o', label=get_method_name(data_type[i]))
    xmin, xmax = plt.xlim()
    plt.hlines(y=y_line, xmin=xmin, xmax=xmax, linewidth=1, colors='gray', linestyles='--', alpha=0.7)
    ax.set_xticks(np.arange(len(data_type)))         
    ax.set_xticklabels([get_method_name(dt) for dt in data_type], rotation=45)
    ax.set_ylabel('Monotonicity Loss')
    ax.set_title('Monotonicity Loss')
    plt.subplots_adjust(bottom=0.3) 
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  
    plt.tight_layout()
    fig_url = os.path.join(save_dir, f"metric_lollipop_monotonicity_loss.pdf")
    plt.savefig(fig_url, dpi=300)   
    plt.close()
        

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
    cur_save_dir = os.path.join(save_dir, "data")
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

     
def main():
    data_type=['blastp', 'diamond', 'mmseq2', 'near', 'dctdomain', 'dhr', 'plm', 'tmvec']
    save_dir = "../results/check_pert_truncation"

    for decoy_type in ['trunchalf', 'truncqrt']:
        for dt in data_type:
            url_origin, url_decoy = build_urls(dt, decoy_type)
            retrieve_data(url_origin, url_decoy, dt, decoy_type, save_dir)
        
    ## qualitative analysis
    plt_compare_stage_joint_density_plot(data_type, save_dir, sample=True)
    
    ## metric
    compare_metric(data_type, save_dir)

    
    
if __name__ == "__main__":
    main()

# command
# python sanity_check/check_pert_truncation.py