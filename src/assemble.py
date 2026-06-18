import pandas as pd
import itertools
import random
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def read_epitope_csv(filepath):
    print(f"INFO: Reading epitopes from '{filepath}'...")
    df = pd.read_csv(filepath)

    # Failsafe: If the user forgets the 'id' column, auto-generate E1, E2, etc.
    if 'id' not in df.columns:
        df['id'] = [f"E{i + 1}" for i in range(len(df))]

    # return [('E1', 'DPLVVAASI'), ('E2', 'LVVAASIIG')]
    return list(zip(df['id'], df['peptide']))


# def generate_linkers(length, total, seed=42, add_linkers=["EAAY", "GPGP", "EAAK"]):
#     print(f"INFO: Generating {total} random linkers of length {length}...")
#     random.seed(seed)
#     AMINOACIDS = "ADEFGHIKLMNPQRSTVWY"
#     unique_linkers = set(add_linkers)
#     while len(unique_linkers) < total + len(add_linkers):
#         linker = "".join(random.choices(AMINOACIDS, k=length))
#         unique_linkers.add(linker)
#     return list(unique_linkers)


def generate_linkers(total_per_length, lengths_list, seed=42, add_controls=True):
    """Generates a stratified pool of linkers across multiple lengths."""
    random.seed(seed)
    AMINOACIDS = "ADEFGHIKLMNPQRSTVWY"
    all_linkers = []

    for length in lengths_list:
        print(f"INFO: Generating {total_per_length} unique linkers of length {length}...")
        unique_linkers = set()

        while len(unique_linkers) < total_per_length:
            linker = "".join(random.choices(AMINOACIDS, k=length))
            unique_linkers.add(linker)
        if add_controls:
            control4 = ["EAAY", "GPGP", "EAAK"]
            control2 = ["GG", "AY"]
            # Standard biological controls to include in the pool
            if length == 4: unique_linkers.update(control4)
            if length == 2: unique_linkers.update(control2)

        all_linkers.extend(list(unique_linkers))

    return all_linkers


def generate_doublets(epitopes, linkers):
    """Generates E_i - L - E_j pairs for ATSP seeding."""
    i = 0
    # Unpack the tuples
    for (id1, e1), (id2, e2) in itertools.permutations(epitopes, 2):
        for linker in linkers:
            seq = e1 + linker + e2

            # (e.g., E9_L4_E9)
            layout = f"E{len(e1)}_L{len(linker)}_E{len(e2)}"

            # ID header (e.g., E1_E2_0|RRQW|E9_L4_E9)
            seq_id = f"{id1}_{id2}_{i}|{linker}|{layout}"

            yield seq_id, seq
            i += 1


def generate_singlets(epitopes, linkers):
    """Generates L - E sequences for N-terminal extension evaluation."""
    i = 0
    for (id_1, epitope) in epitopes:
        for linker in linkers:
            seq = linker + epitope
            layout = f"L{len(linker)}_E{len(epitope)}"
            # e.g., L_E1_0|RRQW|L4_E9
            seq_id = f"L_{id_1}_{i}|{linker}|{layout}"

            yield seq_id, seq
            i += 1


def build_seeding_fasta(input_csv, num_linkers, linker_lens, mode, output_fasta):
    """Main execution function for the assembly module."""
    epitopes = read_epitope_csv(input_csv)

    # e.g., [3, 4, 5]
    linkers = generate_linkers(num_linkers, linker_lens)

    print(f"INFO: Assembly Mode: {mode.upper()}")
    if mode == "doublet":
        generator = generate_doublets(epitopes, linkers)
        total_expected = len(epitopes) * (len(epitopes) - 1) * len(linkers)
    elif mode == "singlet":
        generator = generate_singlets(epitopes, linkers)
        total_expected = len(epitopes) * len(linkers)
    else:
        raise ValueError("Invalid mode selected.")

    print(f"INFO: Generating ~{total_expected} total constructs...")

    records = []
    for seq_id, sequence in generator:
        records.append(SeqRecord(Seq(sequence), id=seq_id, description=""))

    with open(output_fasta, "w") as out_handle:
        SeqIO.write(records, out_handle, "fasta")

    print(f"Seeding generation complete. Saved to {output_fasta}")

# def generate_constructs(epitopes, linkers, structure, epitopes_must_be_different=True, linkers_must_be_different=False):
#     # [Identical to your exact generator function]
#     structure = structure.upper()
#     num_e = structure.count('E')
#     num_l = structure.count('L')
#
#     epitope_iter = itertools.permutations(epitopes, num_e) if epitopes_must_be_different else itertools.product(
#         epitopes, repeat=num_e)
#     linker_iter = itertools.product(linkers, repeat=num_l)
#
#     i = 0
#     for epitope_tuple, linker_tuple in itertools.product(epitope_iter, linker_iter):
#         e_idx, l_idx = 0, 0
#         result_sequence = []
#         for component_type in structure:
#             if component_type == 'E':
#                 result_sequence.append(epitope_tuple[e_idx])
#                 e_idx += 1
#             elif component_type == 'L':
#                 result_sequence.append(linker_tuple[l_idx])
#                 l_idx += 1
#
#         seq_id = f"construct{i + 1}|{linker_tuple[0]}"
#         yield seq_id, "".join(result_sequence)
#         i += 1


# def build_multifasta(input_csv, num_linkers, linker_len, structure, output_fasta):
#     epitopes = read_epitope_csv(input_csv)
#     linkers = generate_linkers(linker_len, num_linkers)
#     generator = generate_constructs(epitopes, linkers, structure)
#
#     count = 0
#     with open(output_fasta, "w") as out:
#         for seq_id, seq_string in generator:
#             record = SeqRecord(Seq(seq_string), id=seq_id, description="")
#             SeqIO.write(record, out, "fasta-2line")
#             count += 1
#     print(f"INFO: Wrote {count} constructs to {output_fasta}")