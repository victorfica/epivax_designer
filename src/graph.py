import pandas as pd
import itertools
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO


def extract_optimal_edges(epitopes, doublets_df):
    """
    Parses the scored doublets to find the single best linker and score
    for every possible directed pair of epitopes (E1 -> E2).
    """
    print("INFO: Mapping optimal linkers to epitope pairs...")
    best_edges = {}

    for _, row in doublets_df.iterrows():
        seq = row['sequence']
        score = row['score']

        # Check all possible directed pairs to see which ones this sequence connects
        for e1, e2 in itertools.permutations(epitopes, 2):
            if seq.startswith(e1) and seq.endswith(e2):
                linker = seq[len(e1):-len(e2)]
                edge_key = (e1, e2)

                # If we haven't seen this pair, or this score is better, update it
                if edge_key not in best_edges or score > best_edges[edge_key]['score']:
                    best_edges[edge_key] = {'linker': linker, 'score': score}

    return best_edges


def assemble_graph(epitopes_csv, scored_doublets_csv, output_fasta):
    """
    Finds the Hamiltonian path through all epitopes that maximizes the total cleavage score.
    """
    # 1. Load the original epitopes
    print(f"INFO: Loading source epitopes from {epitopes_csv}")
    epitopes_df = pd.read_csv(epitopes_csv)
    if 'peptide' not in epitopes_df.columns:
        raise ValueError("Epitope CSV must contain a 'peptide' column.")
    epitopes = epitopes_df['peptide'].tolist()

    # 2. Load the scored doublets
    # Expected columns: 'seq_name', 'sequence', 'score'
    print(f"INFO: Loading scored doublets from {scored_doublets_csv}")
    doublets_df = pd.read_csv(scored_doublets_csv)

    # 3. Build the Adjacency Dictionary (Edges)
    edge_weights = extract_optimal_edges(epitopes, doublets_df)

    # 4. Pathfinding (Brute force permutations for exact global optimum)
    print(f"INFO: Calculating optimal Hamiltonian path for {len(epitopes)} epitopes...")
    best_path = None
    best_total_score = float('-inf')
    best_full_sequence = ""

    # Check every possible ordering of the epitopes
    for path in itertools.permutations(epitopes):
        valid_path = True
        current_score = 0.0
        current_sequence = path[0]

        # Walk the path and sum the edge scores
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i + 1]
            edge = (u, v)

            if edge in edge_weights:
                current_score += edge_weights[edge]['score']
                current_sequence += edge_weights[edge]['linker'] + v
            else:
                # If a doublet was somehow missing from the generation step
                valid_path = False
                break

        if valid_path and current_score > best_total_score:
            best_total_score = current_score
            best_path = path
            best_full_sequence = current_sequence

    # 5. Output Generation
    if best_path is None:
        raise ValueError("Could not find a valid path connecting all epitopes.")

    print("\n--- GRAPH ASSEMBLY COMPLETE ---")
    print(f"Optimal Epitope Order: {' -> '.join([f'E{epitopes.index(e) + 1}' for e in best_path])}")
    print(f"Maximized Processing Score: {best_total_score:.4f}")
    print(f"Draft Sequence: {best_full_sequence}\n")

    # Save to FASTA
    record = SeqRecord(Seq(best_full_sequence), id="graph_assembled_draft|score_optimal", description="")
    with open(output_fasta, "w") as out:
        SeqIO.write(record, out, "fasta")
    print(f"INFO: Saved optimal draft sequence to {output_fasta}")