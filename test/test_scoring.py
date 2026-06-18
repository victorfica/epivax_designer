import numpy as np
from src.scoring import calculate_cleavage_score, calculate_cleavage_score2


def test_scoring_logic():
    # Define a simple architecture: Epitope(9) - Linker(4) - Epitope(9)
    segments = [('Epitope1', 9), ('Linker1', 4), ('Epitope2', 9)]
    total_len = 9 + 4 + 9

    # 1. THE PERFECT CONSTRUCT
    # 0.0 in cores (safe), 0.99 at flanks/linkers (cut)
    perfect_preds = np.zeros(total_len)
    perfect_preds[8] = 0.99  # Cut end of E1
    perfect_preds[9:13] = 0.99  # Cut in Linker
    perfect_preds[13] = 0.99  # Cut start of E2

    # 2. THE SHREDDER
    # 0.99 right in the middle of the epitopes (core destroyed)
    shredder_preds = np.zeros(total_len)
    shredder_preds[4] = 0.99  # Middle of E1 destroyed
    shredder_preds[17] = 0.99  # Middle of E2 destroyed

    # 3. THE BRICK
    # 0.0 everywhere (no cuts)
    brick_preds = np.zeros(total_len)

    # Calculate Scores
    score_perfect = calculate_cleavage_score(perfect_preds, segments)
    score_shredder = calculate_cleavage_score(shredder_preds, segments)
    score_brick = calculate_cleavage_score(brick_preds, segments)

    print(f"Perfect Score:  {score_perfect:.2f} (Should be highest, near 0)")
    print(f"Brick Score:    {score_brick:.2f} (Should be very low/negative)")
    print(f"Shredder Score: {score_shredder:.2f} (Should be the absolute lowest)")


test_scoring_logic()
