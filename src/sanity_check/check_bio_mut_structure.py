import argparse
import os, sys
import pickle
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm import tqdm
import concurrent.futures
from sklearn.metrics import roc_auc_score

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.parser import check_muti_level_homolog, set_randseed, parse_multilvl_homo_result_by_type, get_origin_prot_id, get_seq_label_map, load_sequence, transform_dhr_scores
from utils.metric_utils import run_pairewise_seq_identity_score, run_pairewise_TMscore
from utils.plt_utils import get_method_name, get_color_set

import logging
logging.getLogger("fontTools").setLevel(logging.ERROR)

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

MAX_WORKERS = 28

def build_urls(dt, hit):
    prefix = f'../data/parsed_result/result_{dt}'
    param_prefix = {
            "blastp": "_blosum62_Q11R1",
            "diamond": "_blosum62_Q11R1",
        }.get(dt, "")

    url_origin = f'{prefix}_astral_originrosetta_to_astral{param_prefix}_hit{hit}.txt'
    
    url_mut = f"{prefix}_astral_mutrosetta_to_astral{param_prefix}_hit{hit}.txt" 
    assert os.path.exists(url_origin) and os.path.exists(url_mut)  
    
    return url_origin, url_mut

def find_pdb_path(astral_id, PDBSTYLE_DIR, mut_type=None):
    origin_id = get_origin_prot_id(astral_id)
    if mut_type is None:
        fname = f'{origin_id}.ent'
        pdb_path = os.path.join(PDBSTYLE_DIR, origin_id[2:4], fname)
    else:
        fname = f'{origin_id}_{mut_type}_relax.pdb'
        pdb_path = os.path.join(PDBSTYLE_DIR, origin_id, fname)
    
    return pdb_path

def batch_iterable(iterable, batch_size):
    for i in range(0, len(iterable), batch_size):
        yield iterable[i:i + batch_size]

def construct_remote_homo_pairs(TM_scores, seq_scores, TM_thres=0.5, seq_thres=0.3):
    remote_homo_pairs=[]
    
    for qid in TM_scores:
        for tid in TM_scores[qid]:
            if TM_scores[qid][tid] > TM_thres and seq_scores[qid][tid] < seq_thres:
                remote_homo_pairs.append((qid, tid))
         
    print(f"Find hard pairs: {len(remote_homo_pairs)}")
    return remote_homo_pairs

def extract_similarity_scores(
    url_origin, url_mut, data_type, 
    mut_remote_homo_pairs, origin_remote_homo_pairs, save_path
    ):
    save_url = os.path.join(save_path, f'score_{data_type}.pkl')
    
    if os.path.exists(save_url):
        with open(save_url, 'rb') as f: 
            data = pickle.load(f)  
    else:
        origin_search_map = parse_multilvl_homo_result_by_type(url_origin, data_type) 
        mut_search_map = parse_multilvl_homo_result_by_type(url_mut, data_type) 

        def get_score(map1, map2, qid, tid):
            return (map1.get(qid, {}).get(tid) or map2.get(qid, {}).get(tid))  

        # extract mut remote homo scores
        mut_scores = []
        for qid, tid in tqdm(mut_remote_homo_pairs, total=len(mut_remote_homo_pairs), desc='parsing mut hard pairs'):
            val = get_score(mut_search_map[1], mut_search_map[2], qid, tid)
            if val: 
                mut_score, _ = val
                mut_scores.append(mut_score)
            else:
                # print(f"Skip pair since not in mut query result: qid={qid}, tid={tid}")
                continue

        # extract origin remote homo scores
        origin_scores = []
        for qid, tid in tqdm(origin_remote_homo_pairs, total=len(origin_remote_homo_pairs), desc='parsing mut hard pairs'):
            val = get_score(origin_search_map[1], origin_search_map[2], qid, tid)
            if val: 
                origin_score, _ = val
                origin_scores.append(origin_score)              
            else:
                # print(f"Skip pair since not in query result: qid={qid}, tid={tid}")
                continue               
             
        # extract nonhomo scores
        mut_qid = [qid for qid, _ in mut_remote_homo_pairs]
        all_mut_nonhomo = [
            score
            for k in [-1, 0]
            if k in mut_search_map
            for qid in mut_qid
            for tid, (score, _) in mut_search_map[k][qid].items()
        ]        
        
        origin_qid = [qid for qid, _ in origin_remote_homo_pairs]
        all_origin_nonhomo = [
            score
            for k in [-1, 0]
            if k in origin_search_map
            for qid in origin_qid
            for tid, (score, _) in origin_search_map[k][qid].items()
            if tid == get_origin_prot_id(tid)  # filter out origin → shuffled decoy
        ]
        
        # nonhomo subsample
        def subsample(arr, target_size):
            arr = np.array(arr)
            if len(arr) > target_size:
                indices = np.random.choice(len(arr), size=target_size, replace=False)
                return arr[indices]
            return arr
        
        all_mut_nonhomo_arr = np.array(all_mut_nonhomo)
        all_origin_nonhomo_arr = np.array(all_origin_nonhomo)
        origin_nonhomo_list = [
            subsample(all_origin_nonhomo_arr, len(origin_scores)) for _ in range(20)
        ]
        mut_nonhomo_list = [
            subsample(all_mut_nonhomo_arr, len(mut_scores)) for _ in range(20)
        ]
            
        data = {
            "origin_remote_homo_pair": np.array(origin_scores),
            "mut_remote_homo_pair": np.array(mut_scores),
            # "all_mut_nonhomo_arr": all_mut_nonhomo_arr,
            # "all_origin_nonhomo_arr": all_origin_nonhomo_arr,
            "origin_nonhomo_list": origin_nonhomo_list,
            "mut_nonhomo_list": mut_nonhomo_list,
        }
        print(f"Extracted origin remote homo pair={len(origin_scores)}\tmut remote homo pair={len(mut_scores)}")
        
        with open(save_url, 'wb') as f: 
            pickle.dump(data, f)
        print(f"Saved extracted mutation data to {save_url}")  
          
    return data

def run_batch_TMscore(homo_pairs, src_PDBSTYLE_DIR, tgt_PDBSTYLE_DIR, save_url):
    if os.path.exists(save_url):
        with open(save_url, 'rb') as f:
            TM_scores = pickle.load(f)
        print(f"TM scores already computed and saved at {save_url}")
        
    else:
        print(f"Computing TM scores and saving to {save_url}")
        tasks = {}
        sample_qid = homo_pairs[0][0]
        mut_type = sample_qid[sample_qid.find("mut"):] if "mut" in sample_qid else None
        
        for qid, tid in tqdm(homo_pairs, total=len(homo_pairs), desc='Parsing TMscore...'):
            src_pdb_path = find_pdb_path(qid, src_PDBSTYLE_DIR, mut_type)
            tgt_pdb_path = find_pdb_path(tid, tgt_PDBSTYLE_DIR, mut_type=None)
            
            if get_origin_prot_id(qid) == 'd6a1ia1' or tid == 'd6a1ia1': continue
            tasks[f'{qid}-{tid}'] = (qid, tid, src_pdb_path, tgt_pdb_path)
        print(f"Total tasks: {len(tasks)}")
        
        TM_scores = {}
        tasks = list(tasks.values())
        BATCH_SIZE=100000
        total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE        
        for batch_idx, task_batch in enumerate(batch_iterable(tasks, BATCH_SIZE), start=1):
            print(f"Processing batch {batch_idx}/{total_batches} ...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(run_pairewise_TMscore, t[2], t[3]): t for t in task_batch}
                for future in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures), 
                    desc='TM score computation'
                    ):
                    
                    qid, tid, _, _ = futures[future]
                    
                    try:
                        TM_scores.setdefault(qid, {})[tid] = future.result()
                    except Exception as e:
                        print(f"Error computing TM-score for QRY{qid}_TGT{tid}: {e}")  
        
        # Save results
        with open(save_url, 'wb') as f:
            pickle.dump(TM_scores, f)
    
    return TM_scores      

def run_batch_seq_identity(homo_pairs, src_fasta_url, tgt_fasta_url, save_url):
    if os.path.exists(save_url):
        with open(save_url, 'rb') as f:
            seq_scores = pickle.load(f)
        print(f"seq identity already computed and saved at {save_url}")
        
    else:
        print(f"Computing TM scores and saving to {save_url}")
        seq_scores = {}
        tasks = {}
        src_astral_dic = load_sequence(src_fasta_url) 
        tgt_astral_dic = load_sequence(tgt_fasta_url) 

        for qid, tid in tqdm(homo_pairs, total=len(homo_pairs), desc='Parsing Sequence Identity ...'):
            src_sequence = src_astral_dic[qid]
            tgt_sequence = tgt_astral_dic[tid]
            
            tasks[f'{qid}-{tid}'] = (qid, tid, src_sequence, tgt_sequence)
                
        print(f"Total tasks: {len(tasks)}")
        
        BATCH_SIZE=100000
        seq_scores = {}
        tasks = list(tasks.values())
        total_batches = (len(tasks) + BATCH_SIZE - 1) // BATCH_SIZE     
           
        for batch_idx, task_batch in enumerate(batch_iterable(tasks, BATCH_SIZE), start=1):
            print(f"Processing batch {batch_idx}/{total_batches} ...")
            
            with concurrent.futures.ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
                futures = {executor.submit(run_pairewise_seq_identity_score, t[2], t[3]): t for t in task_batch}
                for future in tqdm(
                    concurrent.futures.as_completed(futures), 
                    total=len(futures), 
                    desc='seqID score computation'
                    ):
                    
                    qid, tid, _, _ = futures[future]
                    
                    try:
                        seqID_score, _ = future.result()
                        seq_scores.setdefault(qid, {})[tid] = seqID_score
                    except Exception as e:
                        print(f"Error computing sequence identity score for QRY{qid}_TGT{tid}: {e}")  

        # Save results
        with open(save_url, 'wb') as f:
            pickle.dump(seq_scores, f)
    
    return seq_scores          
    
def filter_remotely_homologous_pairs(mut_fasta_url, origin_fasta_url, mut_PDBSTYLE_DIR, origin_PDBSTYLE_DIR, TM_thres=0.5, seq_thres=0.3):
    # filter homologous pairs
    mut_astral_dic = load_sequence(mut_fasta_url) 
    origin_astral_dic = load_sequence(origin_fasta_url)
    fa_id_label_map = get_seq_label_map(origin_fasta_url)
    
    homo_pairs = []
    mut_homo_pair = []
    for mut_qid in mut_astral_dic:
        origin_qid = get_origin_prot_id(mut_qid)
        src_labels= fa_id_label_map[origin_qid]
        
        for tid in origin_astral_dic:
            tgt_labels = fa_id_label_map[tid]
            category = check_muti_level_homolog(src_labels, tgt_labels)
            if category in [-1, 0]: continue 
            homo_pairs.append((origin_qid, tid)) 
            mut_homo_pair.append((mut_qid, tid))
    
    dir_name = os.path.join(os.path.dirname(mut_fasta_url), "tmp")
    os.makedirs(dir_name, exist_ok=True)
    
    # input homo_pairs and calculate TM scores
    mut_tmscore_save_url = os.path.join(dir_name, "homo_tmscore_mutrosetta.pkl")
    mut_homo_TM_scores = run_batch_TMscore(mut_homo_pair, mut_PDBSTYLE_DIR, origin_PDBSTYLE_DIR, mut_tmscore_save_url)
    origin_tmscore_save_url = os.path.join(dir_name, "homo_tmscore_origin.pkl")
    origin_homo_TM_scores = run_batch_TMscore(homo_pairs, origin_PDBSTYLE_DIR, origin_PDBSTYLE_DIR, origin_tmscore_save_url)
    
    # input homo_pairs and calculate sequence identity
    mut_seqid_save_url = os.path.join(dir_name, "homo_seqIdentity_mutrosetta.pkl")
    mut_homo_seq_scores = run_batch_seq_identity(mut_homo_pair, mut_fasta_url, origin_fasta_url, mut_seqid_save_url)
    origin_seqid_save_url = os.path.join(dir_name, "homo_seqIdentity_origin.pkl")
    origin_homo_seq_scores = run_batch_seq_identity(homo_pairs, origin_fasta_url, origin_fasta_url, origin_seqid_save_url)            
    
    mut_remote_homo_pairs = construct_remote_homo_pairs(
        mut_homo_TM_scores, mut_homo_seq_scores, 
        TM_thres, seq_thres)
    origin_remote_homo_pairs = construct_remote_homo_pairs(
        origin_homo_TM_scores, origin_homo_seq_scores, 
        TM_thres, seq_thres)  
    
    return mut_remote_homo_pairs, origin_remote_homo_pairs          

def plt_score_distribution(data_type, save_dir, log_scale=True):
    all_dfs = []
    for dt in data_type:
        print(f"Processing data_type={dt}...")
        data_url = os.path.join(save_dir, f"data/score_{dt}.pkl")
        with open(data_url, 'rb') as f: 
            data = pickle.load(f)  
            
        if log_scale:
            # make sure all score > 0
            def add_epsilon(arr):
                arr = np.array(arr)
                return np.array([v + 1e-8 for v in arr if v >= 0])
            data = {
                "origin_remote_homo_pair": add_epsilon(data["origin_remote_homo_pair"]),
                "mut_remote_homo_pair": add_epsilon(data["mut_remote_homo_pair"]),
                # "all_origin_nonhomo_arr": add_epsilon(data["all_origin_nonhomo_arr"]),
                # "all_mut_nonhomo_arr": add_epsilon(data["all_mut_nonhomo_arr"]),
                "origin_nonhomo_list": [add_epsilon(arr) for arr in data["origin_nonhomo_list"]],
                "mut_nonhomo_list": [add_epsilon(arr) for arr in data["mut_nonhomo_list"]],
            }
                    
        if dt == 'dhr':
            data = {
                "origin_remote_homo_pair": transform_dhr_scores(data["origin_remote_homo_pair"]),
                "mut_remote_homo_pair": transform_dhr_scores(data["mut_remote_homo_pair"]),
                # "all_origin_nonhomo_arr": transform_dhr_scores(data["all_origin_nonhomo_arr"]),
                # "all_mut_nonhomo_arr": transform_dhr_scores(data["all_mut_nonhomo_arr"]),                
                "origin_nonhomo_list": [transform_dhr_scores(arr) for arr in data["origin_nonhomo_list"]],
                "mut_nonhomo_list": [transform_dhr_scores(arr) for arr in data["mut_nonhomo_list"]],
            }

        score_map = {
            'origin_remote_homo': data['origin_remote_homo_pair'],
            # 'origin_nonhomo': data["all_origin_nonhomo_arr"],
            'origin_nonhomo': data["origin_nonhomo_list"][0],
            # 'mut_remote_homo': data['mut_remote_homo_pair'],
            'mut_remote_homo': data['mut_nonhomo_list'][0],
            'mut_nonhomo': data["all_mut_nonhomo_arr"]
        }
        df_dt = pd.concat([
            pd.DataFrame({
                'data_type': dt,
                'score': vals,
                'score_type': stype
            })
            for stype, vals in score_map.items()
        ], ignore_index=True)

        all_dfs.append(df_dt)

        # kde plot
        desired_order = ['origin_nonhomo', 'origin_remote_homo', 'mut_nonhomo', 'mut_remote_homo']
        df_dt['score_type'] = pd.Categorical(df_dt['score_type'], categories=desired_order, ordered=True)
        df_dt['source'] = df_dt['score_type'].apply(lambda x: 'origin' if 'origin' in x else 'mut')
        df_dt['pair_type'] = df_dt['score_type'].apply(lambda x: 'remote_homo' if 'remote_homo' in x else 'nonhomo')
                            
        color_set = get_color_set(1)
        style_map = {
            "Remotely homologous (original)": {"color": color_set[0], "linestyle": "-"},
            "Remotely homologous (mutated)": {"color": color_set[0], "linestyle": "--"},
            "Non-homologous (original)": {"color": color_set[1], "linestyle": "-"},
            "Non-homologous (mutated)": {"color": color_set[1], "linestyle": "--"},
        }
        fig_url = os.path.join(save_dir, f"kde{'_log' if log_scale else ''}_{dt}.pdf")
                
        def kdeplot_with_style(x, **kwargs):
            data = kwargs.pop("data")
            score_type = data['score_type'].iloc[0]
            c = color_set[0] if score_type in ['origin_remote_homo', 'mut_remote_homo'] else color_set[1]
            l = '-' if score_type in ['origin_nonhomo', 'origin_remote_homo'] else '--'
            sns.kdeplot(
                data[x],
                color=c,
                linestyle=l,
                fill=True,
                linewidth=1.5,
                clip_on=False,
                bw_adjust=2,
                log_scale=True if log_scale else False,
            )          
        
        sns.set_theme(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
        ax = sns.FacetGrid(df_dt, row="source", hue="pair_type", aspect=2)
        ax.map_dataframe(kdeplot_with_style, "score")
            
        legend_handles = [
            Line2D([0], [0], color=v["color"], linestyle=v["linestyle"], lw=2, label=k)
            for k, v in style_map.items()
        ]
        loc = 'upper right' if dt in ['blastp', 'diamond', 'mmseq2', 'near'] else 'upper left'
        legend = ax.fig.legend(handles=legend_handles, loc=loc, title='', fontsize=12)
        legend.get_frame().set_facecolor("white")
        
        ax.figure.subplots_adjust(hspace=-.2)
        ax.set_titles("")
        ax.set(yticks=[], ylabel="")
        ax.despine(bottom=True, left=True)
        plt.savefig(fig_url)
        plt.close()

def plt_metrics(data_type, data_dir):
    def compute_aucs(pos_scores, neg_scores_list):
        y_true = np.array([1] * len(pos_scores) + [0] * len(neg_scores_list[0]))
        return [roc_auc_score(y_true, np.concatenate([pos_scores, neg])) for neg in neg_scores_list]

    metric_records = []
    for dt in data_type:
        with open(os.path.join(data_dir, f"data/score_{dt}.pkl"), 'rb') as f:
            data = pickle.load(f)
        
        transform = transform_dhr_scores if dt == 'dhr' else lambda x: x
        
        records = [
            {'data_type': dt, 'origin': o, 'mut': m, 'idx': i}
            for i, (o, m) in enumerate(zip(
                compute_aucs(transform(data['origin_remote_homo_pair']), 
                            [transform(arr) for arr in data['origin_nonhomo_list']]),
                compute_aucs(transform(data['mut_remote_homo_pair']), 
                            [transform(arr) for arr in data['mut_nonhomo_list']])
            ))
        ]
        metric_records.extend(records)

    metric_result_df = pd.DataFrame(metric_records)
    metric_result_df['data_type'] = pd.Categorical(
        metric_result_df['data_type'], 
        categories=data_type, ordered=True
    )
    metric_result_df.to_csv(os.path.join(data_dir, f"data/auroc_result.csv"), index=False)

    summary_df = metric_result_df.groupby('data_type')[['origin', 'mut']].agg(['mean', 'std']).round(4)
    summary_df.columns = ['origin_mean', 'origin_std', 'mut_mean', 'mut_std']
    # set data_type order the same as data_type
    
    summary_df.to_csv(os.path.join(data_dir, f"data/metrics.csv"))

    # plot bar chart without error bars
    plt_url = os.path.join(data_dir, f"auroc_bar_plot.pdf")
    plt.figure(figsize=(10, 4))
    
    color_set = get_color_set(2)
    x = np.arange(len(data_type))
    width = 0.35
    
    plt.bar(
        x - width/2, 
        summary_df['origin_mean'], 
        width, 
        label='Original', 
        alpha=0.8,
        color=color_set[0],
        yerr=summary_df['origin_std'],
        capsize=5,
        error_kw={'elinewidth': 2, 'ecolor': 'black'}
    ) 
    
    plt.bar(
        x + width/2, 
        summary_df['mut_mean'], 
        width, 
        label='Mutated', 
        alpha=0.8,
        color=color_set[1],
        yerr=summary_df['mut_std'],
        capsize=5,
        error_kw={'elinewidth': 2, 'ecolor': 'black'}
    )
    
    plt.xticks(x, [get_method_name(dt) for dt in data_type], rotation=45)
    plt.ylim(0.5, 1.01)
    plt.ylabel("AUROC")
    plt.title(f'AUROC Comparison between Origin and Mutated Hard Pairs\n(with mean and standard deviation)')
    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc='upper left',
        facecolor='white'
    )
    plt.tight_layout()
    plt.savefig(plt_url)
    plt.close()
            
        
def main():
    data_type=['blastp', 'diamond', 'mmseq2', 'near', 'dctdomain', 'dhr', 'plm', 'tmvec']
    save_dir = '../results/check_bio_structure'
    
    mut_fasta_url = "../data/db/astral_mutrosetta.fa"
    tgt_fasta_url = "../data/db/astral.fa"
    mut_PDBSTYLE_DIR = "../data/rosetta_mut"
    origin_PDBSTYLE_DIR = "../data/db/astral_pdb/astral40/pdbstyle-2.08"
    
    set_randseed(seed=0)
    
    # construct remote homologous
    mut_remote_homo_pairs, origin_remote_homo_pairs = filter_remotely_homologous_pairs(
        mut_fasta_url, tgt_fasta_url, 
        mut_PDBSTYLE_DIR, origin_PDBSTYLE_DIR, 
        TM_thres=0.5, seq_thres=0.3)

    # extract similarity score
    save_data_dir = os.path.join(save_dir, "data")
    os.makedirs(save_data_dir, exist_ok=True)
    for dt in data_type:
        print(f"Processing {dt}...")
        url_origin, url_mut = build_urls(dt, hit=15177)
        
        extract_similarity_scores(
            url_origin, url_mut, dt, 
            mut_remote_homo_pairs, origin_remote_homo_pairs,
            save_data_dir
            )    

    ## KDE plot
    # Note: For efficient reproduction, we subsample non-homologous targets to match the 
    # number of remote homologous pairs, which yields the same conclusion as using the 
    # full dataset. To reproduce the exact figure from our paper (which uses all non-homologous 
    # targets), replace the subsampled non-homo data with the complete set.
    plt_score_distribution(data_type, save_dir, log_scale=True) 
    
    ## AUROC bar chart
    plt_metrics(data_type, save_dir)

    
            
if __name__ == "__main__":
    main()

# command
# python sanity_check/check_bio_mut_structure.py