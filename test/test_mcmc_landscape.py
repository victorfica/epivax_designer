# Assuming you have your predictor initialized
def test_mcmc_landscape(sequence, predictor, segments):
    base_probs = predictor.predict_sequence(sequence)
    base_score = calculate_cleavage_score(base_probs, segments)

    delta_scores = []
    seq_list = list(sequence)

    # Try 20 random single-point mutations in the linker
    for _ in range(20):
        mutant = seq_list.copy()
        # Mutate the first position of the linker (assuming E1 is 9 AAs)
        mutant[9] = np.random.choice(list("ADEFGHIKLMNPQRSTVWY"))

        mutant_probs = predictor.predict_sequence("".join(mutant))
        mutant_score = calculate_cleavage_score(mutant_probs, segments)

        delta = mutant_score - base_score
        if delta != 0:  # Ignore mutations that didn't change the AA
            delta_scores.append(delta)

    print(f"Average magnitude of ΔS per mutation: {np.mean(np.abs(delta_scores)):.4f}")
    print(f"Max ΔS: {np.max(delta_scores):.4f}, Min ΔS: {np.min(delta_scores):.4f}")

# Run this with one of your draft sequences