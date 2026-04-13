import pandas as pd
import numpy as np
from Bio import SeqIO

EPSILON = 1e-9


def calculate_cleavage_score(cleavage_preds, segments, core_len=9):
    # [Math remains exactly as your original code]
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


def score_constructs(cleavage_csv, fasta_file, output_csv):
    """Calculates scores and saves the ranked constructs to a CSV."""
    print(f"INFO: Scoring predictions from {cleavage_csv}")
    cleavage_df = pd.read_csv(cleavage_csv)
    fasta_constructs = SeqIO.to_dict(SeqIO.parse(fasta_file, "fasta"))

    seq_scores = []
    for protid, protdf in cleavage_df.groupby("prot_id"):
        # Assuming the doublets are E(9)-L(4)-E(9) based on your assembly setup
        segments = [('Epitope1', 9), ('Linker1', 4), ('Epitope2', 9)]
        vax_process_score = calculate_cleavage_score(protdf.prediction.values, segments)
        seq_scores.append((protid, vax_process_score))

    # Sort by highest score
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