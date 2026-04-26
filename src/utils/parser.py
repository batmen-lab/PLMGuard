import sys, os, logging, pickle
import numpy as np
from tqdm import tqdm
from Bio import SeqIO
import argparse
from pathlib import Path
import pathlib
import random
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir))

def get_origin_prot_id(prot_id):
    if '_range' in prot_id: prot_id = prot_id[:prot_id.find('_range')]
    if '_rev' in prot_id: prot_id = prot_id[:prot_id.find('_rev')]
    if '_doub_shuf0' in prot_id: prot_id = prot_id[:prot_id.find('_doub_shuf0')]
    if '_doub_shuf1' in prot_id: prot_id = prot_id[:prot_id.find('_doub_shuf1')]
    if '_decoy' in prot_id: prot_id = prot_id[:prot_id.find('_decoy')]
    if '_trunc' in prot_id: prot_id = prot_id[:prot_id.find('_trunc')]
    if '_doub' in prot_id: prot_id = prot_id[:prot_id.find('_doub')]
    if '_mut' in prot_id: prot_id = prot_id[:prot_id.find('_mut')]
    if '_shuf' in prot_id: prot_id = prot_id[:prot_id.find('_shuf')]
    if '_mkv' in prot_id: prot_id = prot_id[:prot_id.find('_mkv')]
    if '_cfpgen' in prot_id: prot_id = prot_id[:prot_id.find('_cfpgen')]
    if '_len' in prot_id: prot_id = prot_id[:prot_id.find('_len')]

    return prot_id

def get_seq_label_map(fasta_url):
    records = list(SeqIO.parse(fasta_url, "fasta"))

    fa_id_label_map = {}
    for record in tqdm(records):
        fa_id = record.id
        desc = record.description
        scop_label = desc.split(' ')[1]
        arr = scop_label.split('.')
        assert len(arr) == 4

        scop_fold = '.'.join(arr[:2])
        scop_supf = '.'.join(arr[:3])
        scop_fam = '.'.join(arr[:4])

        assert fa_id not in fa_id_label_map
        fa_id_label_map[fa_id] = (scop_fold, scop_supf, scop_fam)

    return fa_id_label_map

def check_homolog(src_scop_fold, src_scop_supf, tgt_scop_fold, tgt_scop_supf):
    if src_scop_supf == tgt_scop_supf: return 1
    elif src_scop_fold != tgt_scop_fold: return -1
    else: return 0

def check_muti_level_homolog(src_labels, tgt_labels):
    '''
        -1: non-homolog
        0: same fold, different superfamily
        1: same superfamily, different family
        2: same family
    '''
    src_scop_fold, src_scop_supf, src_scop_fam = src_labels
    tgt_scop_fold, tgt_scop_supf, tgt_scop_fam = tgt_labels
    
    if src_scop_fold != tgt_scop_fold:
        category = -1              
    elif src_scop_supf != tgt_scop_supf:
        category = 0
    elif src_scop_fam != tgt_scop_fam:
        category = 1
    else:
        category = 2
    return category 
    
def parse_plm_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            arr = line.split()
            res.append([arr[0], arr[1], float(arr[-1])])
    return res

def parse_tmvec_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            if line.startswith('query_id'): continue
            
            arr = line.split()
            res.append([arr[0], arr[2], float(arr[-1])])
    return res

def parse_dhr_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            
            arr = line.split(',')
            res.append([arr[0], arr[1], float(arr[-1])])
    return res

def parse_dctdomain_data(url):
    res = []
    curr_tgt_set = set()
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            if not line.startswith('Query'): continue

            arr = line.split(':')
            assert len(arr) == 4

            curr_id = arr[1].strip().split()[0]
            tgt_id = arr[2].strip().split()[0]
            score = float(arr[3].strip())
            
            # dctdomain may contain duplicate entries, we only keep the first one
            if (curr_id, tgt_id) not in curr_tgt_set: 
                curr_tgt_set.add((curr_id, tgt_id))
                res.append([curr_id, tgt_id, score])
    return res

def parse_blast_data(url):
    curr_query_id = ''
    score_zone = False
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue

            if line.startswith("Results from"):
                curr_iter = int(line.split()[-1])
                print('curr_iter={}'.format(curr_iter))
                assert curr_iter <= 1

            elif line.startswith("Query="):
                curr_query_id = line.split()[1]

            elif line.startswith("Sequences"): score_zone = True
            elif line.startswith("Lambda"): score_zone = False
            else:
                if not score_zone: continue
                data = line.split()
                sid = data[0]
                evalue = float(data[-1])
                raw_bit = float(data[-2])
                
                yield (curr_query_id, sid, raw_bit, evalue)

def parse_mmseq2_data(url):
    res = []
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            
            arr = line.split()
            res.append([arr[0], arr[1], float(arr[3]), float(arr[2])])
    return res

def parse_near_data(url):
    assert url.endswith('.csv')
    
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            qid, tid, score = line.split('\t')
            
            yield (qid, tid, float(score))

def parse_diamond_data(url):
    res = []
    assert url.endswith('.tsv')
    
    with open(url, 'r') as fp:
        for cnt, line in enumerate(fp):
            line = line.rstrip().rstrip("\n")
            if len(line) <= 0: continue
            
            arr = line.split()
            res.append([arr[0], arr[1], float(arr[3]), float(arr[4])])
    return res

ParserFunctionMap = {
    "plm": parse_plm_data,
    "tmvec": parse_tmvec_data,
    "dhr": parse_dhr_data,
    "dctdomain": parse_dctdomain_data,
    "blastp": parse_blast_data,
    "mmseq2": parse_mmseq2_data,
    "near": parse_near_data,
    "diamond": parse_diamond_data
}

def parse_score_by_type(input_path, data_type):
    assert data_type in ParserFunctionMap.keys()
    
    fname = Path(input_path).stem
    parent_parent_dir = Path(input_path).parent.parent
    save_url = os.path.join(f"{parent_parent_dir}/parsed_pkl", f'{fname}.pkl')
    os.makedirs(Path(save_url).parent, exist_ok=True)
    print(f"saved to {save_url}")

    if os.path.exists(save_url):
        print(f"File already exists: {save_url}, loading...")
        with open(save_url, 'rb') as f:
            qt_score = pickle.load(f)  
            
        qt_score_homo = qt_score['homo']
        qt_score_nonhomo = qt_score['nonhomo']
        
    else:
        assert os.path.exists(input_path)
        with open(input_path, 'r') as f:
            parsed_result = f.readlines()
            
        qt_score_homo={}
        qt_score_nonhomo={}
        
        for row in tqdm(parsed_result[1:], desc='Sparsing batch...'):
            curr_id, tgt_id, score, homo_type, rank, *rest = row.strip().split('\t')
            score, homo_type, rank = float(score), int(homo_type), int(rank)
            
            curr_id_origin = get_origin_prot_id(curr_id)
            if curr_id not in qt_score_homo: qt_score_homo[curr_id] = []
            if curr_id not in qt_score_nonhomo: qt_score_nonhomo[curr_id] = []     
                   
            if homo_type > 0: 
                qt_score_homo[curr_id].append((curr_id_origin, tgt_id, score))
            else:
                qt_score_nonhomo[curr_id].append(score)
                
        qt_score = {'homo': qt_score_homo, 'nonhomo': qt_score_nonhomo}

        with open(save_url, 'wb') as fp:
            pickle.dump(qt_score, fp)
            
    print('sid_search_map_homo={}\tsid_search_map_nonhomo={}'.format(len(qt_score_homo), len(qt_score_nonhomo)))
    return qt_score_homo, qt_score_nonhomo

def parse_multilvl_homo_result_by_type(input_path, data_type):
    assert data_type in ParserFunctionMap.keys()
    script_dir = pathlib.Path(__file__).parent.absolute()
    fasta_url = script_dir / "../../data/db/astral.fa"
    fasta_url = fasta_url.resolve()

    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))

    fname = Path(input_path).stem
    parent_parent_dir = Path(input_path).parent.parent
    save_url = os.path.join(f"{parent_parent_dir}/parsed_pkl", f'{fname}_ml.pkl')
    print(f"saved to {save_url}")

    if os.path.exists(save_url):
        print(f"File already exists: {save_url}, loading...")
        with open(save_url, 'rb') as f:
            qt_result = pickle.load(f)  
                    
    else:
        assert os.path.exists(input_path), f"Input path= {input_path}"
        with open(input_path, 'r') as f:
            parsed_result = f.readlines()
            
        qt_result = {
            -1: defaultdict(dict),
            0: defaultdict(dict),
            1: defaultdict(dict),
            2: defaultdict(dict),
        }
        
        for row in tqdm(parsed_result[1:], desc='Sparsing batch...'):
            curr_id, tgt_id, score, homo_type, rank, *rest = row.strip().split('\t')
            score, homo_type, rank = float(score), int(homo_type), int(rank)
            qt_result[homo_type][curr_id][tgt_id] = (score, rank)
        
        with open(save_url, 'wb') as fp:
            pickle.dump(qt_result, fp)
    
    print('fam_homo={}\tsupf_homo={}\tfold_homo={}\tnonhomo={}'.format(
        len(qt_result[2]), len(qt_result[1]), len(qt_result[0]), len(qt_result[-1])
        ))
    
    return qt_result

def parse_search_result_by_type(input_path, save_path, data_type, fasta_url):
    skip_max_hits_check = ['dctdomain', 'blastp', 'psiblast', 'mmseq2', 'diamond', 'near']
    skip_score_order_check = ['blastp', 'psiblast', 'mmseq2', 'diamond']
    files_w_evalue = ['blastp', 'psiblast', 'mmseq2', 'diamond']
    result_file_types = ('.out', '.txt', '.tsv', '.csv')
    
    assert data_type in ParserFunctionMap.keys()

    fa_id_label_map = get_seq_label_map(fasta_url)
    print('fa_id_label_map={}'.format(len(fa_id_label_map)))
    
    fname = Path(input_path).name
    save_url = os.path.join(save_path, f'parsed_result/{fname}_hit{max_hits}.txt')

    if os.path.exists(save_url):
        with open(save_url, 'r') as f:
            qt_score = f.readlines()
        print(f"Parsed results already exists (line={len(qt_score)}): {save_url}!")    
                     
    else:
        assert os.path.exists(input_path)
        os.makedirs(os.path.dirname(save_url), exist_ok=True)
        
        parse_func = ParserFunctionMap.get(data_type)
        
        with open(save_url, 'w') as f:
            if data_type in files_w_evalue:
                f.write(f"qid\ttid\tscore\thomo_type\trank\tevalue\n")
            else:
                f.write(f"qid\ttid\tscore\thomo_type\trank\n")
        
        total_line = 0
        for file in tqdm(os.listdir(input_path), desc='Sparsing batch...'):
            if not file.endswith(result_file_types):  continue
            
            url = os.path.join(input_path, file)
            print('url={}'.format(url))

            query_id_flag = None
            rank = None
            prev_score = None # for sanity check
            for row in tqdm(parse_func(url), desc='Sparsing file...'):
                curr_id, tgt_id, score, *rest = row
                score = float(score)
                
                # get ranking
                if query_id_flag != curr_id:
                    if rank is not None and data_type not in skip_max_hits_check: 
                        assert rank >= max_hits, f"Error: rank {rank} < max_hits {max_hits} for query {curr_id}!"
                    query_id_flag = curr_id
                    rank = 1
                    prev_score = score
                else:
                    rank += 1
                    if data_type not in skip_score_order_check:
                        assert score >= prev_score if data_type == 'dhr' else score <= prev_score
                    prev_score = score
                    
                if rank > max_hits: continue
                
                # get homo type
                curr_id_origin = get_origin_prot_id(curr_id)
                assert curr_id_origin in fa_id_label_map, f"Error: {curr_id_origin} not found in label map! original id {curr_id}"
                
                tgt_id_origin = get_origin_prot_id(tgt_id)
                assert tgt_id_origin in fa_id_label_map, f"Error: {tgt_id_origin} not found in label map!"
                
                src_labels = fa_id_label_map[curr_id_origin]
                tgt_labels = fa_id_label_map[tgt_id_origin]
                
                homo_type = check_muti_level_homolog(src_labels, tgt_labels)
                
                # save results
                with open(save_url, 'a') as f:
                    if rest:
                        f.write(f"{curr_id}\t{tgt_id}\t{score}\t{homo_type}\t{rank}\t{rest[0]}\n")
                    else:
                        f.write(f"{curr_id}\t{tgt_id}\t{score}\t{homo_type}\t{rank}\n")
                total_line += 1

        print(f"Parsed results (line={total_line}) saved to: {save_url}.")   

def set_randseed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

def load_sequence(fasta_url):
    astral_dic = {}
    for record in SeqIO.parse(fasta_url, "fasta"):
        astral_dic[record.id] = str(record.seq)
    return astral_dic

def transform_dhr_scores(s, delta_t=150):
    """
    Transform DHR scores:
    score = -score + delta_t
    """
    trans_s = -1 * s + delta_t
    assert (np.array(trans_s) > 0).all()

    return trans_s

    
max_hits=None
def main(args):
    global max_hits
    max_hits = args.max_hits

    script_dir = pathlib.Path(__file__).parent.absolute()
    fasta_url = script_dir / "../../data/db/astral.fa"    
    parse_search_result_by_type(args.input_path, args.save_path, args.data_type, fasta_url)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parser for PLMs")
    parser.add_argument("--data_type", type=str, help="PLMs method")
    parser.add_argument("--input_path", type=str, help="Input path to be parsed")
    parser.add_argument("--save_path", type=str, help="Path to save parsed result")
    parser.add_argument("--max_hits", type=int, default=1000, help="Keep how many hits while parsing")

    args = parser.parse_args()
    main(args)

# command
# python utils/parser.py --data_type blastp --basedb SCOPe40 --input_path ../data/temp/result_blastp_astral_blosum62_Q11R1 --save_path ../data