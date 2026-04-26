from Bio import pairwise2 as pw2
from Bio.pairwise2 import format_alignment
import subprocess
import re, os

def run_pairewise_seq_identity_score(sequence1, sequence2):
    global_align = pw2.align.globalxx(sequence1, sequence2)
    best_sequence_identity = -1
    align = "None"
    for i in global_align:
        sequence_identity = i[2]/(i[4]-i[3])
        if (sequence_identity > best_sequence_identity):
            align = i
            best_sequence_identity = sequence_identity
    return best_sequence_identity, format_alignment(*align)


def run_pairewise_TMscore(pdb_path1, pdb_path2):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tm_score_exe = os.path.abspath(os.path.join(current_dir, '../../libs/TMscore'))
    
    output = subprocess.run(
        [tm_score_exe, pdb_path1, pdb_path2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    result = output.stdout
    
    match = re.findall(r"TM-score\s*=\s*([\d\.]+)", result)
    assert len(match) == 1, f"Expected one TM-score, found {len(match)} in result: {result}."
    tm_scores = float(match[0])
    
    return tm_scores  