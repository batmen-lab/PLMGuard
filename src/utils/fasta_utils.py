import os, sys, argparse, math, logging, subprocess, random
from collections import defaultdict
sys.path.append('..')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import numpy as np
from Bio import SeqIO
from tqdm import tqdm


def fasta_to_upper(fasta_url, output_url):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        out_file.write(">{}\n".format(desc))
        line_num = math.ceil(len(seq) / 60)
        for i in range(line_num):
            out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def fasta_to_tsv(fasta_url, output_url):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        # desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))
        out_file.write("{}\t{}\n".format(ID, seq))

    out_file.close()

def fasta_mutation(fasta_url, output_url, model='WAG'):
    import pyvolve
    tree = pyvolve.read_tree(tree="(((t1:0.36,t2:0.45):0.001,t3:0.77):0.44,(t5:0.77,t4:0.41):0.89);", scale_tree=2.0)
    ### root > t1 > t2 > t3 > t4 > t5
    assert model in ['JTT', 'WAG', 'LG', 'DAYHOFF', 'MTMAM', 'MTREV24']
    m = pyvolve.Model(model)

    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description

        if 'B' in seq: seq = seq.replace('B', '')
        if 'J' in seq: seq = seq.replace('J', '')
        if 'Z' in seq: seq = seq.replace('Z', '')
        if 'X' in seq: seq = seq.replace('X', '')
        if 'U' in seq: seq = seq.replace('U', '')
        if 'O' in seq: seq = seq.replace('O', '')

        p = pyvolve.Partition(root_sequence=str(seq), models=m)
        evolve = pyvolve.Evolver(partitions=p, tree=tree)
        evolve(ratefile=False, infofile=False, seqfile=False)
        seqdict = evolve.get_sequences(anc=True)

        for key in seqdict:
            if 'internal' in key or 'root' in key: continue

            sim_seq = seqdict[key]
            out_file.write(">{}_mut_{}\n".format(ID, key))
            line_num = math.ceil(len(sim_seq) / 60)
            for i in range(line_num):
                out_file.write("{}\n".format(sim_seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def fasta_doublen(fasta_url, output_url, shuffle=True):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = np.random.default_rng(0)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description

        seq_list = list(seq)
        if shuffle: rng.shuffle(seq_list)
        append_seq = "".join(seq_list)
        concat_seq = seq + append_seq

        out_file.write(">{}_doub_shuf{}\n".format(ID, int(shuffle)))
        line_num = math.ceil(len(concat_seq) / 60)
        for i in range(line_num):
            out_file.write("{}\n".format(concat_seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def fasta_trunclen(fasta_url, output_url, ratio, replicate=1):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))
    percent = int(ratio * 100)

    rng = np.random.default_rng(0)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq.upper()
        desc = record.description

        seqlen = len(seq)
        reduced_seqlen = int(seqlen * ratio)
        start_indices = np.random.randint(0, (seqlen - reduced_seqlen - 1), replicate)
    
        for idx in range(replicate):
            start_idx = start_indices[idx]
            reduced_seq = seq[start_idx: (start_idx+reduced_seqlen)]
            out_file.write(">{}_trunc_{}pct_rep{}\n".format(ID, percent, idx))

            line_num = math.ceil(len(reduced_seq) / 60)
            for i in range(line_num):
                out_file.write("{}\n".format(reduced_seq[60 * i: 60 * (i + 1)]))

    out_file.close()

def fasta_genshuf(fasta_url, output_url, include_origin=False):
    assert os.path.exists(fasta_url)
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    rng = np.random.default_rng(0)
    for record in tqdm(records):
        ID = record.id
        seq = record.seq
        desc = record.description
        # print('ID={}\tdesc={}'.format(ID, desc))

        seq_list = list(seq)
        rng.shuffle(seq_list)
        shuf_seq = "".join(seq_list)
        assert len(seq) == len(shuf_seq)
        line_num = math.ceil(len(seq) / 60)

        if include_origin:
            out_file.write(">{}\n".format(ID))
            for i in range(line_num):
                out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

        out_file.write(">{}_shuf\n".format(ID))
        for i in range(line_num):
            out_file.write("{}\n".format(shuf_seq[60 * i: 60 * (i + 1)]))

    out_file.close()

 
def main(args):
    assert os.path.exists(args.fasta_url)
    
    if args.type == 'mutant':
        ### generate mutant sequences (Experiment 1: Evolutionary plausibility)
        output_url = args.fasta_url.replace('.fa', '_mutantWAG.fa')
        fasta_mutation(args.fasta_url, output_url, model='WAG')
        fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    
        # output_url = args.fasta_url.replace('.fa', '_mutantJTT.fa')
        # fasta_mutation(args.fasta_url, output_url, model='JTT')
        # fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))    
    elif args.type == 'doublen':
        ### generate double self and double shuf sequences (Experiment 3: Redundancy stability)
        output_url = args.fasta_url.replace('.fa', '_doubshuf.fa')
        fasta_doublen(args.fasta_url, output_url, shuffle=True)
        fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
        
        output_url = args.fasta_url.replace('.fa', '_doubself.fa')
        fasta_doublen(args.fasta_url, output_url, shuffle=False)
        fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    elif args.type == 'trunclen':
        ### generate truncated sequences (Experiment 4: Similarity monotonicity)
        output_url = args.fasta_url.replace('.fa', '_truncqrt.fa')
        fasta_trunclen(args.fasta_url, output_url, ratio=0.25, replicate=1)
        fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
        
        output_url = args.fasta_url.replace('.fa', '_trunchalf.fa')
        fasta_trunclen(args.fasta_url, output_url, ratio=0.5, replicate=1)
        fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    elif args.type == 'shuf':
        ### generate shuf sequence (Experiment 5: Data manipulation safety )
        output_url = args.fasta_url.replace('.fa', '_shuf.fa')
        fasta_genshuf(args.fasta_url, output_url)
        fasta_to_tsv(output_url, output_url.replace('.fa', '.tsv'))
    elif args.type == 'to_tsv':
        fasta_to_tsv(args.fasta_url, args.fasta_url.replace('.fa', '.tsv'))
    else:
        raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Optional app description')
    parser.add_argument('--fasta_url', type=str, help='fasta_url')
    parser.add_argument('--type', type=str, help='fasta_url')

    main(parser.parse_args())

### command
# python utils/fasta_utils.py --fasta_url ../data/astral.fa
