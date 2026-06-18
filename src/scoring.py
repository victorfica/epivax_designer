import pandas as pd
import numpy as np
from Bio import SeqIO

EPSILON = 1e-9


def calculate_cleavage_score(cleavage_preds, segments, core_len=9):

    score_protection = 0.0
    score_processing = 0.0
    current_pos = 0

    for seg_name, seg_len in segments:
        if seg_len == 0: continue
        seg_preds = cleavage_preds[current_pos: current_pos + seg_len]

        if 'epitope' in seg_name.lower():
            core_offset = (seg_len - core_len) // 2
            core_preds = seg_preds[core_offset: core_offset + core_len - 1]
            score_protection += np.log(1 - core_preds + EPSILON).sum()

            flank_preds = np.concatenate([
                seg_preds[0: core_offset],
                seg_preds[core_offset + core_len - 1: seg_len]
            ])
            if flank_preds.size > 0:
                score_processing += np.log(flank_preds.max() + EPSILON)
        else:
            if seg_preds.size > 0:
                score_processing += np.log(seg_preds.max() + EPSILON)

        current_pos += seg_len
    return score_protection + score_processing


def calculate_cleavage_score2(cleavage_preds, segments):
    """
    Calculates the additive log-likelihood processing score of a multi-epitope construct.

    Equation:
    S = \sum \ln(\max P_l) + \sum [ \ln(P_anchor) + \sum \ln(1 - P_c) ]
    """
    score_protection = 0.0
    score_processing = 0.0
    current_pos = 0

    for seg_name, seg_len in segments:
        if seg_len == 0:
            continue

        seg_preds = cleavage_preds[current_pos: current_pos + seg_len]

        if 'epitope' in seg_name.lower():
            # Penalize: Cleavage inside epitope
            # Everything BEFORE the final C-terminal residue.
            core_preds = seg_preds[:-1]
            if core_preds.size > 0:
                score_protection += np.log(1 - core_preds + EPSILON).sum()

            # Reward: Cleavage precisely at C-terminal
            # Strictly isolates the exact final residue of the epitope sequence.
            p_anchor = seg_preds[-1]
            score_processing += np.log(p_anchor + EPSILON)

        elif 'linker' in seg_name.lower():
            # 3. Processing Reward: The Spacer (Liberate N-terminus for ERAP trimming)
            # Rewards the single highest cleavage probability anywhere inside the linker.
            if seg_preds.size > 0:
                score_processing += np.log(seg_preds.max() + EPSILON)

        current_pos += seg_len

    return score_protection + score_processing

# def score_constructs(cleavage_csv, fasta_file, output_csv):
#     """Calculates scores and saves the ranked constructs to a CSV."""
#     print(f"INFO: Scoring predictions from {cleavage_csv}")
#     cleavage_df = pd.read_csv(cleavage_csv)
#     fasta_constructs = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))
#
#     seq_scores = []
#     for protid, protdf in cleavage_df.groupby("prot_id"):
#         # Assuming the doublets are E(9)-L(4)-E(9) based on your assembly setup
#         segments = [('Epitope1', 9), ('Linker1', 4), ('Epitope2', 9)]
#         vax_process_score = calculate_cleavage_score2(protdf.prediction.values, segments)
#         seq_scores.append((protid, vax_process_score))
#
#     # Sort by highest score
#     construct_scored = sorted(seq_scores, key=lambda x: x[1], reverse=True)
#
#     # Extract sequence and format for output
#     results_to_save = []
#     for seqname, score in construct_scored:
#         sequence = str(fasta_constructs[seqname].seq)
#         results_to_save.append({
#             'seq_id': seqname,
#             'sequence': sequence,
#             'score': score
#         })
#
#     out_df = pd.DataFrame(results_to_save)
#     out_df.to_csv(output_csv, index=False)
#     print(f"INFO: Scored {len(out_df)} constructs and saved ranked list to {output_csv}")
#     return output_csv


def score_constructs(cleavage_csv, fasta_file, output_csv):
    """Calculates scores and saves the ranked constructs to a CSV dynamically."""
    print(f"INFO: Scoring predictions from {cleavage_csv}")
    cleavage_df = pd.read_csv(cleavage_csv)
    fasta_constructs = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    seq_scores = []
    for protid, protdf in cleavage_df.groupby("prot_id"):

        # 1. Parse the layout dynamically from the FASTA ID
        # Format expectation: E1_E2_6467|RRQW|E9_L4_E9
        parts = str(protid).split('|')

        if len(parts) >= 3:
            layout_str = parts[2]
            segments = []

            # Rebuild the segment tuples dynamically: [('Epitope_0', 9), ('Linker_1', 4)]
            for idx, part in enumerate(layout_str.split('_')):
                comp_type = part[0] # 'E' or 'L'
                comp_len = int(part[1:])  # '9' -> 9

                name = f"Epitope{idx}" if comp_type == 'E' else f"Linker{idx}"
                segments.append((name, comp_len))
        else:
            # Fallback for old fasta files generated before the update
            print(f"WARNING: No layout metadata found for {protid}. Falling back to default 9-4-9.")
            segments = [('Epitope1', 9), ('Linker1', 4), ('Epitope2', 9)]

        # 2. Pass the dynamic segments to the mathematically corrected scoring function
        vax_process_score = calculate_cleavage_score2(protdf.prediction.values, segments)
        seq_scores.append((protid, vax_process_score))

    # Sort by highest score (closest to 0)
    construct_scored = sorted(seq_scores, key=lambda x: x[1], reverse=True)

    # Extract sequence and format for output
    results_to_save = []
    for seqname, score in construct_scored:
        sequence = str(fasta_constructs[seqname].seq)
        results_to_save.append({
            'seq_id': seqname,
            'sequence': sequence,
            'score': score
        })

    out_df = pd.DataFrame(results_to_save)
    out_df.to_csv(output_csv, index=False)
    print(f"INFO: Scored {len(out_df)} constructs and saved ranked list to {output_csv}")
    return output_csv