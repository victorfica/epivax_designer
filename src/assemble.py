import pandas as pd
import itertools
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def read_epitope_csv(filepath):
    print(f"INFO: Reading epitopes from '{filepath}'...")
    df = pd.read_csv(filepath)
    return df['peptide'].tolist()


def generate_linkers(length, total, seed=42, add_linkers=["EAAY", "GPGP", "EAAK"]):
    print(f"INFO: Generating {total} random linkers of length {length}...")
    random.seed(seed)
    AMINOACIDS = "ADEFGHIKLMNPQRSTVWY"
    unique_linkers = set(add_linkers)
    while len(unique_linkers) < total + len(add_linkers):
        linker = "".join(random.choices(AMINOACIDS, k=length))
        unique_linkers.add(linker)
    return list(unique_linkers)


def generate_constructs(epitopes, linkers, structure, epitopes_must_be_different=True, linkers_must_be_different=False):
    # [Identical to your exact generator function]
    structure = structure.upper()
    num_e = structure.count('E')
    num_l = structure.count('L')

    epitope_iter = itertools.permutations(epitopes, num_e) if epitopes_must_be_different else itertools.product(
        epitopes, repeat=num_e)
    linker_iter = itertools.product(linkers, repeat=num_l)

    i = 0
    for epitope_tuple, linker_tuple in itertools.product(epitope_iter, linker_iter):
        e_idx, l_idx = 0, 0
        result_sequence = []
        for component_type in structure:
            if component_type == 'E':
                result_sequence.append(epitope_tuple[e_idx])
                e_idx += 1
            elif component_type == 'L':
                result_sequence.append(linker_tuple[l_idx])
                l_idx += 1

        seq_id = f"construct{i + 1}|{linker_tuple[0]}"
        yield seq_id, "".join(result_sequence)
        i += 1


def build_multifasta(input_csv, num_linkers, linker_len, structure, output_fasta):
    epitopes = read_epitope_csv(input_csv)
    linkers = generate_linkers(linker_len, num_linkers)
    generator = generate_constructs(epitopes, linkers, structure)

    count = 0
    with open(output_fasta, "w") as out:
        for seq_id, seq_string in generator:
            record = SeqRecord(Seq(seq_string), id=seq_id, description="")
            SeqIO.write(record, out, "fasta-2line")
            count += 1
    print(f"INFO: Wrote {count} constructs to {output_fasta}")