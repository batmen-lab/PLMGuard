import pandas as pd
import argparse
from Bio import SeqIO
import os

def load_fasta_ids(fasta_path):
    id_to_seqid = {}
    for (i, record) in enumerate(SeqIO.parse(fasta_path, "fasta")):
        id_to_seqid[i+1] = record.id
    return id_to_seqid

def convert_near_data(url, query_id_dic, target_id_dic, max_hits=-1):
    df = pd.read_csv(
        url, sep=' ', header=None, 
        names=['query_id', 'target_id', 'score'],
        dtype={"query_id": int, "target_id": int, "score": float})
    
    # sort by id ascending, then by score descending
    df = df.sort_values(by=['query_id', 'score'], ascending=[True, False])
    
    if max_hits > 0:
        df = df.groupby('query_id').head(max_hits)
            
    df['query_id'] = df['query_id'].map(query_id_dic)
    df['target_id'] = df['target_id'].map(target_id_dic)

    basename = os.path.basename(url)
    save_url = url.replace(basename, 'sorted_hits.csv')
    df.to_csv(save_url, sep='\t', index=False, header=False)

def main(args):
    query_id_dic = load_fasta_ids(args.query_fasta)
    target_id_dic = load_fasta_ids(args.target_fasta)
    convert_near_data(args.input_path, query_id_dic, target_id_dic, args.max_hits)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert Near output to desired format")
    parser.add_argument("--input_path", type=str, help="Input CSV file path from Near output")
    parser.add_argument("--query_fasta", type=str, help="Path to query fasta file")
    parser.add_argument("--target_fasta", type=str, help="Path to target fasta file")
    parser.add_argument("--max_hits", type=int, default=-1, help="Keep how many hits while parsing, -1 means all")

    args = parser.parse_args()
    main(args)  

# command example:
# python utils/convert_near.py --input_path /PATH/data/temp/result_near_astral_to_astral/hits.csv
# --query_fasta /PATH/data/db/astral.fa
# --target_fasta /PATH/data/db/astral.fa