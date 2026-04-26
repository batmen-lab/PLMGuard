import argparse, sys, os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.metrics import roc_auc_score

import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parser import load_sequence, parse_multilvl_homo_result_by_type, set_randseed, transform_dhr_scores
from utils.plt_utils import get_method_name, get_color_set

sample_keys = {
    'plm': ['d2cxaa1', 'd7ms7a_', 'd1b35b_', 'd3nv7a_', 'd3weoa4', 'd1qjta_'],
    'tmvec': ['d2cxaa1', 'd7ms7a_', 'd1b35b_', 'd3nv7a_', 'd3weoa4', 'd1qjta_'],
    'dhr': ['d2bykb1', 'd2cxaa1','d7ms7a_', 'd1b35b_', 'd3nv7a_', 'd3weoa4', 
            'd1qjta_', 'd1i1ra1', 'd6v6fb_', 'd6dcma_', 'd3mxta1'],
    'dctdomain': ['d4aana1', 'd3cx5d1', 'd2jn6a1', 'd1igna2', 'd2cxaa1', 'd7ms7a_', 
                  'd1b35b_', 'd3nv7a_', 'd3weoa4', 'd1qjta_'],
}

def build_urls(dt, decoy_type):
    prefix = f"../data/parsed_result/result"

    origin_url = f"{prefix}_{dt}_astral_to_astral_shuf_ori_hit30354.txt" 
    perm_url =  f"{prefix}_{decoy_type}_{dt}_astral_to_astral_hit15177.txt" 
    assert os.path.exists(origin_url), f"File not found: {origin_url}" 
    assert os.path.exists(perm_url), f"File not found: {perm_url}" 
    
    return origin_url, perm_url

def plt_auroc(fasta_url, save_dir):
    data_type=['dctdomain', 'dhr', 'plm', 'tmvec']
    decoy_type='perm'
    
    auroc_df_url = f"{save_dir}/data/metrics.csv"
    if os.path.exists(auroc_df_url):
        auroc_df = pd.read_csv(auroc_df_url)
    else:
        os.makedirs(os.path.dirname(auroc_df_url), exist_ok=True)
        auroc_data = []
        sample_keys = load_sequence(fasta_url) 
        for dt in data_type:   
            origin_url, perm_url = build_urls(dt, decoy_type)
            origin_search_map = parse_multilvl_homo_result_by_type(origin_url, dt)
            perm_search_map = parse_multilvl_homo_result_by_type(perm_url, dt)  
                      
            aucroc_list = []
            for qid in tqdm(sample_keys, total=len(sample_keys), desc=f"Retrieve auc data for {dt}..."):
                score = []
                label = []
                for homo_type in perm_search_map:
                    for tid, (s, rank) in perm_search_map[homo_type][qid].items():
                        label.append(0)
                        score.append(s)
                
                for homo_type in origin_search_map:
                    for tid, (s, rank) in origin_search_map[homo_type][qid].items():
                        if 'shuf' in tid: continue
                        
                        label.append(1)
                        score.append(s)
                                                
                score_arr = np.asarray(score)
                label_arr = np.asarray(label)
                
                if dt == 'dhr':
                    score_arr = transform_dhr_scores(score_arr)
                
                aucroc = roc_auc_score(label_arr, score_arr)
                aucroc_list.append(aucroc)
                auroc_data.append({
                    'data_type': dt,
                    'query_id': qid,
                    'auroc': aucroc,
                })
                

        auroc_df = pd.DataFrame(auroc_data)
        auroc_df.to_csv(auroc_df_url, index=False)
        print(f"Saved AUROC data to {auroc_df_url}")
    
    # plot violin plot
    color = get_color_set()
    color = color[4:]
    plt.figure(figsize=(4, 4))
    ax = sns.violinplot(
        data=auroc_df,
        x='data_type',
        y='auroc',
        palette=color,
    )
    plt.axhline(y=0.5, color='gray', linestyle='--', linewidth=1)
    
    ax.set_xticklabels(
        [get_method_name(t.get_text()) for t in ax.get_xticklabels()]
    )
    plt.ylim(-0.06, 1.06)
    plt.xlabel('')
    plt.ylabel('AUROC')
    plt.title('AUROC Distribution by Method')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"auroc_violin_{decoy_type}.pdf"))
    plt.close()

def plt_dist_by_query(save_dir): 
    data_dir = os.path.join(save_dir, "data")
    cur_save_path = os.path.join(save_dir, "KDE") 
    decoy_type = 'perm'
    
    for dt in sample_keys.keys():
        origin_url, perm_url = build_urls(dt, decoy_type)
                
        data_local_url = f"{data_dir}/{decoy_type}_{dt}.npy"
        os.makedirs(os.path.dirname(data_local_url), exist_ok=True)

        origin_search_map = parse_multilvl_homo_result_by_type(origin_url, dt)
        perm_search_map = parse_multilvl_homo_result_by_type(perm_url, dt)
        
        keys = sample_keys[dt]
        parsed_data_local = retrieve_data_local(origin_search_map, perm_search_map, keys, dt)
        print(f"Plotting distribution for {dt}...")
        plt_distribution(
            parsed_data_local, dt, decoy_type, show_homo=True, 
            save_path=f"{cur_save_path}/{dt}", 
            ) 
        keys = sample_keys[dt]


def plt_distribution(parsed_data_local, data_type, decoy_type, save_path, logscale=True, show_homo=True):
    os.makedirs(save_path, exist_ok=True)
    for qid in parsed_data_local:
        fname = f"{'w' if show_homo else 'wo'}homo_{decoy_type}_{data_type}_{qid}{'_log' if logscale else ''}.pdf"
        fig_url = os.path.join(save_path, fname)
        
        homo = parsed_data_local[qid]['homo']
        nonhomo = parsed_data_local[qid]['nonhomo']
        decoy = parsed_data_local[qid]['decoy']

        if data_type == 'dhr': 
            homo = transform_dhr_scores(homo)
            nonhomo = transform_dhr_scores(nonhomo)
            decoy = transform_dhr_scores(decoy)

        plt.figure(figsize=(10, 5))
        sns.kdeplot(nonhomo, label='Non-homologous', fill=True, linewidth=1)
        sns.kdeplot(decoy, label='Decoy', fill=True, linewidth=1)
        
        if show_homo:
            for i, s in enumerate(homo):
                if i == 0:
                    plt.axvline(x=s, color='r', linestyle='--', label='Homologous', alpha=0.7, linewidth=1)
                else:
                    plt.axvline(x=s, color='r', linestyle='--', alpha=0.7, linewidth=1)
            
        if logscale:
            plt.xscale('log')
                                          
        plt.title(f"KDE of {get_method_name(data_type)}")
        plt.xlabel('Score')
        plt.ylabel('Density')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left') 
        plt.tight_layout()  
        plt.savefig(fig_url)
        plt.close()        
        
def retrieve_data_local(origin_search_map, perm_search_map, sampled_keys, data_type):
    use_similarity = True
    if data_type == 'dhr': use_similarity = False
    
    parsed_data_local = {}
    
    for origin_qid in sampled_keys:
        homo_score = []
        nonhomo_score = []
        decoy_score = []
        
        # homo & decoy
        for homo_type in [1, 2]:
            for tid, (score, rank) in origin_search_map[homo_type][origin_qid].items():
                if 'shuf' in tid: continue
                homo_score.append(score)
                    
        # nonhomo & decoy    
        for homo_type in [-1, 0]:
            for tid, (score, rank) in origin_search_map[homo_type][origin_qid].items():
                if 'shuf' in tid: continue
                nonhomo_score.append(score)  
                
        for homo_type in  perm_search_map:
            for tid, (score, rank) in perm_search_map[homo_type][origin_qid].items():
                decoy_score.append(score)
        
        homo_score_arr = np.asarray(homo_score)
        nonhomo_score_arr = np.asarray(nonhomo_score)
        decoy_score_arr = np.asarray(decoy_score)

        if use_similarity:
            homo_score_arr = np.sort(homo_score_arr)[::-1]
            nonhomo_score_arr = np.sort(nonhomo_score_arr)[::-1]
            decoy_score_arr = np.sort(decoy_score_arr)[::-1]
        else:
            homo_score_arr = np.sort(homo_score_arr)
            nonhomo_score_arr = np.sort(nonhomo_score_arr)
            decoy_score_arr = np.sort(decoy_score_arr)
            
        min_len = min(len(nonhomo_score_arr), len(decoy_score_arr))
        if min_len == 0: 
            print(f"{origin_qid} has {len(nonhomo_score_arr)} nonhomo and {len(decoy_score_arr)} decoy")
        else:
            parsed_data_local[origin_qid] = {
                'homo': homo_score_arr,
                'nonhomo': nonhomo_score_arr,
                'decoy': decoy_score_arr
            }

    
    return parsed_data_local

def qqplot(save_dir): 
    cur_save_path = os.path.join(save_dir, "qqplot_norm_rank")
    data_dir = os.path.join(save_dir, "data")
    decoy_type = 'perm'
    
    for dt in sample_keys.keys():
        print(f"Processing {dt}")
        origin_url, perm_url = build_urls(dt, decoy_type)
        origin_search_map = parse_multilvl_homo_result_by_type(origin_url, dt)
        perm_search_map = parse_multilvl_homo_result_by_type(perm_url, dt)  
                
        keys = sample_keys[dt]
        for qid in keys:
            qid_query_url = os.path.join(f"{data_dir}/{dt}", f"{qid}.csv")
            
            if os.path.exists(qid_query_url):
                query_result_df = pd.read_csv(qid_query_url)
            else:
                os.makedirs(os.path.dirname(qid_query_url), exist_ok=True)
                query_result = []
                for homo_type in perm_search_map:
                    for tid, (s, rank) in perm_search_map[homo_type][qid].items():
                        if dt == 'dhr':
                            s = transform_dhr_scores(s)
                            
                        label="decoy"
                        is_true_target=False

                        query_result.append({
                            "tid": tid,
                            "score": s,
                            "label": label,
                            "is_true_target": is_true_target
                        })                            

                for homo_type in origin_search_map:
                    for tid, (s, rank) in origin_search_map[homo_type][qid].items():
                        if 'shuf' in tid: continue
                        
                        if dt == 'dhr':
                            s = transform_dhr_scores(s)
                                                                                                
                        if homo_type in [1, 2]:
                            label="target"
                            is_true_target=True
                        else:
                            label="target"
                            is_true_target=False
                        query_result.append({
                            "tid": tid,
                            "score": s,
                            "label": label,
                            "is_true_target": is_true_target
                        })
                        
                query_result_df = pd.DataFrame(query_result)
                query_result_df = query_result_df.sort_values("score", ascending=False).reset_index(drop=True)
                
                query_result_df.to_csv(qid_query_url, index=False)
                
            # figure
            fig_url = os.path.join(cur_save_path, f"{dt}/rank_scatter_{qid}.pdf")
            os.makedirs(os.path.dirname(fig_url), exist_ok=True)
            n_total = len(query_result_df)
            rank = np.arange(n_total) + 1
            query_result_df["rank_norm"] = (rank + 1) / (n_total + 1)
            
            is_target = (query_result_df["label"] == "target").astype(int).values
            is_decoy  = (query_result_df["label"] == "decoy").astype(int).values
            is_true_target = query_result_df["is_true_target"].values                
            is_self_homo = (
                (query_result_df["is_true_target"] == True) &
                (query_result_df["tid"] == qid)
            ).values
            is_other_homo = (
                (query_result_df["is_true_target"] == True) &
                (query_result_df["tid"] != qid)
            ).values
                
            rank_norm = query_result_df["rank_norm"].values

            target_rank = np.where(is_target, rank_norm, 0.0)
            decoy_rank  = np.where(is_decoy,  rank_norm, 0.0)

            max_target_rank = np.maximum.accumulate(target_rank)
            max_decoy_rank  = np.maximum.accumulate(decoy_rank)
            # ---- FIX LOG(0) ----
            eps = 1.0 / (n_total + 1)
            max_target_rank = np.maximum(max_target_rank, eps)
            max_decoy_rank  = np.maximum(max_decoy_rank,  eps)
            
            plt.figure(figsize=(5,5))
            plt.scatter(
                max_target_rank[~is_true_target],
                max_decoy_rank[~is_true_target],
                s=10,
                alpha=0.3,
                rasterized=True,
                label="Others"
            )

            plt.scatter(
                max_target_rank[is_self_homo],
                max_decoy_rank[is_self_homo],
                s=80,
                marker='*',
                color='red',
                zorder=4,
                label="Self target"
            )

            plt.scatter(
                max_target_rank[is_other_homo],
                max_decoy_rank[is_other_homo],
                s=80,
                marker='*',
                color='orange',
                zorder=3,
                label="Homologous target"
            )
            
            # ideal diagonal
            maxv = max(max_target_rank.max(), max_decoy_rank.max())
            plt.plot([1e-6, maxv], [1e-6, maxv], '--', color='gray')

            plt.xscale('log')
            plt.yscale('log')

            eps = 1e-6
            plt.xlim(eps, maxv)
            plt.ylim(eps, maxv)

            plt.xlabel("Max target normalized rank")
            plt.ylabel("Max decoy normalized rank")
            plt.title("Ranking Comparison QQ Plot")

            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_url, dpi=300)
            plt.close()



def main():
    save_dir = "../results/check_perm_model"
    set_randseed(seed=0)
    
    ## Metric
    fasta_url = "../data/db/astral.fa"
    plt_auroc(fasta_url, save_dir)
    
    # Case study
    plt_dist_by_query(save_dir)
    qqplot(save_dir)    
    
if __name__ == "__main__":
    main()

# command
# python sanity_check/check_perm_model.py