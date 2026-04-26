from pyrosetta import *
from pyrosetta.rosetta import *
from pyrosetta.rosetta.protocols.relax import FastRelax
from pyrosetta.rosetta.protocols.simple_moves import MutateResidue

import random
import math
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import concurrent.futures

from Bio import SeqIO
from Bio.PDB import PDBParser
from Bio.Data.IUPACData import protein_letters_3to1

from metric_utils import run_pairewise_seq_identity_score, run_pairewise_TMscore

pyrosetta.init("-ignore_unrecognized_res true -load_PDB_components false")

# Expand 3to1 mapping to handle common equivalent residues (e.g. MSE→M)
THREE_TO_ONE = dict(protein_letters_3to1)
THREE_TO_ONE.update({
    "MSE": "M",  # Replace MSE (SelenoMethionine) with M
})

THREE_TO_ONE = {k.upper(): v for k, v in THREE_TO_ONE.items()}

STANDARD_AA_1 = set("ACDEFGHIKLMNPQRSTVWY")
STANDARD_AA_3 = {k for k, v in THREE_TO_ONE.items() if v in STANDARD_AA_1}

def extract_sequence_from_pdb(pdb_path):
    """Extract sequences per chain (1-letter) from a PDB file."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    sequences = {}
    for model in structure:
        for chain in model:
            seq = ""
            for residue in chain:
                hetfield, resseq, icode = residue.id
                if hetfield != " ":continue
                
                resname = residue.get_resname().upper().strip()
                if resname in THREE_TO_ONE:
                    seq += THREE_TO_ONE[resname]
            sequences[chain.id] = seq
        break
    
    key = list(sequences.keys())
    assert len(key) == 1, f"Sequences: {sequences}"
    
    return sequences[key[0]]

def relax_pose(pose):
    relax = FastRelax()
    scorefxn = get_fa_scorefxn()
    relax.set_scorefxn(scorefxn)
    relax.constrain_relax_to_start_coords(False)
    # print(relax)
    relax.apply(pose)
    return pose

def pack(pose, posi, amino, scorefxn):
    '''
    Source: https://nbviewer.org/github/RosettaCommons/PyRosetta.notebooks/blob/master/notebooks/06.08-Point-Mutation-Scan.ipynb
    '''
    print("Mutating residue", posi, pose.residue(posi).name1(), "to", amino)
    # Select Mutate Position
    mut_posi = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_posi.set_index(posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(mut_posi.apply(pose)))

    # Select Neighbor Position
    nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_selector.set_focus_selector(mut_posi)
    nbr_selector.set_include_focus_in_subset(True)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(nbr_selector.apply(pose)))

    # Select No Design Area
    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_posi)
    #print(pyrosetta.rosetta.core.select.get_residues_from_subset(not_design.apply(pose)))

    # The task factory accepts all the task operations
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()

    # These are pretty standard
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Disable Packing
    prevent_repacking_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    prevent_subset_repacking = pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking_rlt, nbr_selector, True )
    tf.push_back(prevent_subset_repacking)

    # Disable design
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(),not_design))

    # Enable design
    aa_to_design = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_to_design.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(aa_to_design, mut_posi))
    
    # Create Packer
    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover()
    packer.task_factory(tf)

    #Perform The Move
    if not os.getenv("DEBUG"):
      packer.apply(pose)

def mutate_residue(pose, posi, amino, pack_radius=8.0, repack=True):
    '''
    https://docs.rosettacommons.org/docs/latest/scripting_documentation/RosettaScripts/Movers/movers_pages/MutateResidueMover
    '''
    aa_map = {
            'A': 'ALA', 'C': 'CYS', 'D': 'ASP', 'E': 'GLU',
            'F': 'PHE', 'G': 'GLY', 'H': 'HIS', 'I': 'ILE',
            'K': 'LYS', 'L': 'LEU', 'M': 'MET', 'N': 'ASN',
            'P': 'PRO', 'Q': 'GLN', 'R': 'ARG', 'S': 'SER',
            'T': 'THR', 'V': 'VAL', 'W': 'TRP', 'Y': 'TYR'
        }

    scorefxn = get_fa_scorefxn()
    
    # mutate residue
    mutator = MutateResidue(posi, aa_map[amino])
    mutator.apply(pose)

    if repack:
        # TaskFactory to repack surrounding residues
        tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

        # repack only residues within pack_radius of the mutation site
        mut_res = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector(str(posi))
        nbr_selector = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
        nbr_selector.set_focus_selector(mut_res)
        # nbr_selector.set_focus(posi)
        nbr_selector.set_distance(pack_radius)
        nbr_selector.set_include_focus_in_subset(True)
        
        prevent_repacking = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
        tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(prevent_repacking, nbr_selector, flip_subset=True))
        
        # apply PackRotamersMover
        packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
        packer.task_factory(tf)
        packer.apply(pose)
    
    return pose

def load_mut_pos(mut_list_path, mut_line=1):
    mut_list = {}
    with open(mut_list_path) as f:
        lines = f.readlines()
        parse_line = lines[mut_line-1].strip().replace(";", "")
        for s in parse_line.split(","):
            posi = int(s[2:-1])
            amino = s[-1]
            chain = s[1]
            mut_list[posi] = (chain, amino)
    return mut_list
        
# SOFT LINK the repaired PDBs to the output dir
def soft_link_pdbs(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(src):
        if os.path.exists(dst):
            os.remove(dst)
        os.symlink(src, dst)
        print(f"Linked {src} -> {dst}")
    else:
        print(f"PDB not found for linking: {src}", file=sys.stderr)

def read_astral(fasta_url):    
    astral_map = {}
    for record in SeqIO.parse(fasta_url, "fasta"):
        astral_map[record.id] = str(record.seq)
    return astral_map

def load_mutable_positions(pdb_path):
    """Return a list of mutable positions: [ (chain_id, resseq, aa1) , ... ]
    Only standard amino acids are included, excluding HETATM and unknown residues.
    resseq is an integer 
    (insertion code is ignored, may lose precision if there are insertion codes).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)

    mutable = []

    for model in structure:
        for chain in model:
            cid = chain.id
            for residue in chain:
                hetfield, resseq, icode = residue.id
                if hetfield != " ":
                    # exclude HETATM/water etc.
                    continue
                resname = residue.get_resname().upper().strip()
                if resname not in THREE_TO_ONE:
                    continue
                aa1 = THREE_TO_ONE[resname]
                if aa1 not in STANDARD_AA_1:
                    continue
                
                try:
                    resseq_int = int(resseq)
                    mutable.append((cid, resseq_int, aa1))
                except ValueError:
                    print(f"Non-integer resseq: {resseq}, skip")
                    continue
                
        break  # Keep only the first model

    return mutable

def pick_targets(mutable, k, rng):
    k = max(0, min(k, len(mutable)))
    return rng.sample(mutable, k)

def choose_new_aa(orig, rng):
    choices = [a for a in STANDARD_AA_1 if a != orig]
    return rng.choice(choices)

def to_mutline(positions, rng):
    """Convert a list of positions to a FoldX mutline 
    (single line, comma-separated, ending with semicolon).
    E.g.: AA100G,GB200F;
    """
    parts = []
    for cid, resseq, aa1 in positions:
        newaa = choose_new_aa(aa1, rng)
        parts.append(f"{aa1}{cid}{resseq}{newaa}")
    return ",".join(parts) + ";\n"

def gen_single_mut(pdb_path, mut_pos_outdir, rng):
    mutfile = os.path.join(mut_pos_outdir, f"individual_list.txt")
    
    if not os.path.exists(mutfile):
        print(f"Generating new mutfile: {mutfile}")
        mutable = load_mutable_positions(pdb_path)
        if not mutable:
            print("Mutateable positions not found (standard amino acids). Check if PDB contains standard residues.", file=sys.stderr)
            # sys.exit(1)
            return False
            
        L = len(mutable)

        rand_positions = pick_targets(mutable, L, rng)
        rand_mutline = to_mutline(rand_positions, rng)
        
        os.makedirs(os.path.dirname(mutfile), exist_ok=True)
        with open(mutfile, "w") as f:
            f.write(rand_mutline)

        print(f"Generate mutfile:\n  {mutfile}: {rand_mutline.strip()}")

    mut_list = load_mut_pos(mutfile, mut_line=1)    
    return mut_list    

def run_rosetta_sequence(seq, pdb_path, save_dir, rng, mut_type=1, mut_pct=60):
    # mutation position
    mut_list = gen_single_mut(pdb_path, save_dir, rng)
    
    # first relax the pdb
    pose = pose_from_pdb(pdb_path)
    relax_path = pdb_path.replace('.pdb', '_relax.pdb')
    if not os.path.exists(relax_path):
        relaxPose = relax_pose(pose)
        relaxPose.dump_pdb(relax_path)
    else:
        relaxPose = pose_from_pdb(relax_path)
    
    # run mutation
    k = max(1, int(mut_pct * len(mut_list) / 100))
    
    sequence1 = extract_sequence_from_pdb(relax_path)
    
    n=1
    result_df = []
    scorefxn = get_score_function()
    mutPose = relaxPose.clone()
        
    for posi, (chain, amino) in mut_list.items(): 
        try:
            print(f"Mutate chain {chain} position {posi} to {amino}")  
            pose_num = pose.pdb_info().pdb2pose(chain, posi)
            print("Mutating residue", pose_num, pose.residue(pose_num).name1(), f"to {amino}") 
            if mut_type == 1:
                pack(mutPose, pose_num, amino, scorefxn)
            elif mut_type == 2:
                mutate_residue(mutPose, pose_num, amino, pack_radius=8.0, repack=True)
            # print("\nNew Energy:", scorefxn(relaxPose),"\n")
        except Exception as e:
            print(f"Error mutating chain {chain} position {posi} to {amino}: {e}")
            continue
        
        if n == k:
            mut_path = pdb_path.replace('.pdb', f'_mut{mut_pct}p.pdb')
            mutrelax_path = pdb_path.replace('.pdb', f'_mut{mut_pct}p_relax.pdb')
            mutPose.dump_pdb(mut_path)
            
            # further Relax
            mutrelaxPose = relax_pose(mutPose)
            mutrelaxPose.dump_pdb(mutrelax_path)
            print(f"Saved mutation {mut_pct}% to {mut_path} and {mutrelax_path}")
            
            sequence2 = extract_sequence_from_pdb(mutrelax_path)
            
            TM_score1 = run_pairewise_TMscore(relax_path, mutrelax_path)
            TM_score2 = run_pairewise_TMscore(pdb_path, mutrelax_path)
            seq_identity1 = run_pairewise_seq_identity_score(sequence1, sequence2)[0]
            seq_identity2 = run_pairewise_seq_identity_score(seq, sequence2)[0]
            result_df.append({
                "mut_type": mut_type,
                "mut_percent": mut_pct,
                "seq": seq,
                "seq_relax": sequence1,
                "seq_mutrelax": sequence2,
                "TM_score (relax vs mutrelax)": TM_score1,
                "TM_score (origin vs mutrelax)": TM_score2,
                "seq_identity (relax vs mutrelax)": seq_identity1,
                "seq_identity (origin vs mutrelax)": seq_identity2,
            })
            break
        else:
            n += 1

    return result_df

def process_sequence(seq_id, sequence, pdb_dir, save_dir, rng, mut_type=1, mut_pct=60):
    print(f"Processing {seq_id}, length {len(sequence)}")
    
    src_ent_url = os.path.abspath(os.path.join(pdb_dir, seq_id[2:4], f"{seq_id}.ent"))
    dst_pdb_url = os.path.abspath(os.path.join(save_dir, seq_id, f"{seq_id}.pdb"))
    soft_link_pdbs(src_ent_url, dst_pdb_url)
    seq_save_dir = os.path.join(save_dir, seq_id)
    
    try:
        result = run_rosetta_sequence(sequence, dst_pdb_url, seq_save_dir, rng, mut_type, mut_pct)
        result_df = pd.DataFrame(result)
        result_df.to_csv(os.path.join(seq_save_dir, f"mut_summary.csv"), index=False)
        
        new_entries = {seq_id: sequence}
        for line in result:
            new_id = f"{seq_id}_mut{mut_pct}p"
            new_entries[new_id] = line['seq_mutrelax']
            
        return new_entries
    
    except Exception as e:
        print(f"Error processing {seq_id}: {e}")
        return {seq_id: sequence}

def check_existing_pdb(astral_map, pdb_dir):
    for seq_id, sequence in tqdm(astral_map.items(), desc="Processing sequences"):
        src_ent_url = os.path.abspath(os.path.join(pdb_dir, seq_id[2:4], f"{seq_id}.ent"))
        assert os.path.exists(src_ent_url), f"PDB file not found for {seq_id} at {src_ent_url}"
    print("Pass check: All PDB files exist!")  
      
def main(pdb_dir, fasta_url, save_dir, rng):
    fname = os.path.basename(fasta_url).replace('.fa', '_mutrosetta.fa')
    mut_fasta_url = os.path.join(save_dir, fname)
    astral_map = read_astral(fasta_url)
            
    new_astral = {}
    max_workers = 20
    mut_type=1
    mut_pct = 60
    print(f"Using {max_workers} workers for parallel processing")
    
    check_existing_pdb(astral_map, pdb_dir)   
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_id = {
            executor.submit(process_sequence, seq_id, sequence, pdb_dir, save_dir, rng, mut_type, mut_pct): seq_id 
                        for seq_id, sequence in astral_map.items()}     

        for future in tqdm(concurrent.futures.as_completed(future_to_id), total=len(future_to_id), desc="Processing sequences"):
            seq_id = future_to_id[future]
            try:
                result = future.result()
                new_astral.update(result)
            except Exception as e:
                print(f"Error processing {seq_id}: {e}")
                # new_astral[seq_id] = astral_map[seq_id]
                                   
    with open(mut_fasta_url, "w") as f:
        for seq_id, seq in new_astral.items():
            if f"_mut{mut_pct}p" not in seq_id: 
                continue
            
            f.write(f">{seq_id}\n")
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + "\n")        

def subsample_fasta(fasta_url, sample_number=1000):
    assert os.path.exists(fasta_url)
    output_url = str(Path(fasta_url).parent / f"astral_s{sample_number}.fa")
    out_file = open(output_url, "w")
    records = list(SeqIO.parse(fasta_url, "fasta"))

    random.seed(0)
    random.shuffle(records)
    count_selected = 0

    for record in tqdm(records):
        if count_selected > sample_number: 
            break
        
        ID = record.id
        # filter sequence w/o pdb file   
        if ID.startswith('g1'): continue
        
        seq = record.seq.upper()
        desc = record.description
        count_selected += 1
        
        line_num = math.ceil(len(seq) / 60)
        
        out_file.write(">{}\n".format(ID))
        for i in range(line_num):
            out_file.write("{}\n".format(seq[60 * i: 60 * (i + 1)]))

    out_file.close()
    print('Subsampled {} sequences from {}'.format(count_selected, fasta_url))
 
                 
    
if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    lib_path = Path(current_dir).parent.parent
    
    pdb_dir = f"{lib_path}/data/db/astral_pdb/astral40/pdbstyle-2.08"
    fasta_url = f"{lib_path}/data/db/astral.fa"
    save_dir = f"{lib_path}/data/rosetta_mut"
    os.makedirs(save_dir, exist_ok=True)
    
    rng = random.Random(0)
    subsample_fasta(fasta_url, sample_number=1000)
    
    main(pdb_dir, fasta_url, save_dir, rng)
    

    
### Command  
# python utils/run_mut_rosetta.py
            