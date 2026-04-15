import os
import random
import math
import tempfile
import numpy as np
import torch
import pandas as pd
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord

# Import the core logic from your decoupled package
from mhcicleave import config
from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
from mhcicleave.cleavagedatasets import MultiFastaWindowDataset
from mhcicleave.encodings import Embedder

# Import your validated scoring logic
from src.scoring import calculate_cleavage_score


class DynamicPredictor:
    """
    A lightweight, high-speed wrapper around the mhcicleave model.
    Designed specifically to evaluate single sequences rapidly during the MCMC loop.
    """

    def __init__(self, checkpoint=None, device_str="cuda"):
        self.device = torch.device(device_str if torch.cuda.is_available() else "cpu")

        # Fallback to the packaged default checkpoint if none is provided
        if checkpoint is None:
            checkpoint = config.DEFAULT_CHECKPOINT
            if checkpoint is None:
                raise ValueError("No checkpoint provided and no default bundled model found.")

        print(f"INFO: Initializing DL model for MCMC from {checkpoint}")

        # Initialize Architecture
        mlp_network = MLP(**config.DEFAULT_MLP_PARAMS)
        self.model = CleavageLightningModule.load_from_checkpoint(checkpoint, model=mlp_network)
        self.model.to(self.device)
        self.model.eval()  # Crucial: Set to evaluation mode (disables dropout, etc.)

        # Initialize Embedder
        self.embedder = Embedder(name=config.ESM2_MODEL_NAME, device=self.device)

    def predict_sequence(self, sequence: str) -> np.ndarray:
        """
        Dynamically extracts windows, generates ESM-2 embeddings, and predicts cleavage.
        Uses a temporary FASTA to flawlessly integrate with your existing dataset classes.
        """
        # 1. Create a rapid temp file for the dataset parser
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.fasta') as tmp:
            tmp.write(f">mcmc_candidate\n{sequence}\n")
            tmp_path = tmp.name

        try:
            # 2. Extract windows using your package logic
            window_dataset = MultiFastaWindowDataset(
                fasta_file=tmp_path,
                upstream=config.UPSTREAM,
                downstream=config.DOWNSTREAM
            )

            # 3. Generate Embeddings
            # Setting num_workers=0 is faster here since we only have one sequence
            seq_embeddings = self.embedder.encode_esm(
                window_dataset,
                emb_type="concat",
                batch_size=len(window_dataset),
                num_workers=0
            )

            # 4. Rapid Forward Pass (Bypassing Lightning Trainer overhead)
            with torch.no_grad():
                # Pass embeddings directly to the underlying MLP
                tensor_preds = self.model(seq_embeddings.to(self.device))
                probabilities = tensor_preds.flatten().cpu().numpy()

            return probabilities

        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


class MCMCOptimizer:
    """
    Executes the Simulated Annealing logic to refine linker sequences in-situ.
    """

    def __init__(self, predictor, epitope_len=9, linker_len=4, initial_temp=2.0, cooling_rate=0.99):
        self.predictor = predictor
        self.epitope_len = epitope_len
        self.linker_len = linker_len
        self.T = initial_temp
        self.cooling_rate = cooling_rate
        self.AMINOACIDS = "ADEFGHIKLMNPQRSTVWY"  # Excluding Cysteine based on your earlier logic

    def _build_segments(self, sequence: str):
        """Dynamically maps the E-L-E-L-E structure based on sequence length."""
        segments = []
        curr_len = 0
        seq_len = len(sequence)
        count = 1

        while curr_len < seq_len:
            segments.append((f'Epitope{count}', self.epitope_len))
            curr_len += self.epitope_len
            if curr_len < seq_len:
                segments.append((f'Linker{count}', self.linker_len))
                curr_len += self.linker_len
            count += 1
        return segments

    def _get_mutable_indices(self, segments):
        """Identifies which amino acid positions correspond to linkers."""
        mutable_indices = []
        curr_pos = 0
        for seg_name, seg_len in segments:
            if 'Linker' in seg_name:
                mutable_indices.extend(range(curr_pos, curr_pos + seg_len))
            curr_pos += seg_len
        return mutable_indices

    def calculate_energy(self, sequence: str, segments: list) -> float:
        """Energy is the negative processing score (since MCMC minimizes energy)."""
        probs = self.predictor.predict_sequence(sequence)
        score = calculate_cleavage_score(probs, segments, core_len=self.epitope_len)
        return -score

    def propose_move(self, sequence: str, mutable_indices: list) -> str:
        """Performs a single random point mutation within the linker regions."""
        if not mutable_indices:
            return sequence  # Safety catch if no linkers exist

        seq_list = list(sequence)
        target_idx = random.choice(mutable_indices)
        current_aa = seq_list[target_idx]

        # Pick a new amino acid that is different from the current one
        new_aa = random.choice([aa for aa in self.AMINOACIDS if aa != current_aa])
        seq_list[target_idx] = new_aa

        return "".join(seq_list)

    def optimize(self, initial_sequence: str, iterations: int) -> tuple:
        """The main Simulated Annealing loop."""
        segments = self._build_segments(initial_sequence)
        mutable_indices = self._get_mutable_indices(segments)

        current_seq = initial_sequence
        current_energy = self.calculate_energy(current_seq, segments)

        best_seq = current_seq
        best_energy = current_energy

        print(f"INFO: Starting MCMC Optimization (Temp: {self.T}, Iterations: {iterations})")
        print(f"INFO: Initial Cleavage Score (S): {-current_energy:.4f}")

        for i in range(iterations):
            # 1. Propose a new sequence
            new_seq = self.propose_move(current_seq, mutable_indices)
            new_energy = self.calculate_energy(new_seq, segments)

            # 2. Metropolis Acceptance Criterion
            delta_E = new_energy - current_energy

            # If the score improves (energy drops), or if the thermal noise allows it
            if delta_E < 0 or random.random() < math.exp(-delta_E / self.T):
                current_seq = new_seq
                current_energy = new_energy

                # Keep track of the absolute best sequence found
                if current_energy < best_energy:
                    best_seq = current_seq
                    best_energy = current_energy

            # 3. Cool down the system
            self.T *= self.cooling_rate

            # Optional: Print progress every 200 iterations
            if (i + 1) % 200 == 0:
                print(
                    f"      Step {i + 1}/{iterations} | Current Score (S): {-current_energy:.4f} | Temp: {self.T:.4f}")

        print(f"INFO: MCMC Complete. Best Optimized Score (S): {-best_energy:.4f}")
        return best_seq, -best_energy


def run_mcmc_optimization(
        initial_fasta: str,
        checkpoint: str = None,
        iterations: int = 1000,
        epitope_len: int = 9,
        linker_len: int = 4
):
    """
    CLI wrapper function to execute the MCMC optimization pipeline.
    """
    records = list(SeqIO.parse(initial_fasta, "fasta"))
    if not records:
        print(f"ERROR: No sequences found in {initial_fasta}")
        return

    # We take the first sequence from the draft FASTA
    seq_id = records[0].id
    initial_seq = str(records[0].seq)

    # Initialize the engine
    predictor = DynamicPredictor(checkpoint=checkpoint)
    optimizer = MCMCOptimizer(predictor, epitope_len=epitope_len, linker_len=linker_len)

    # Run the optimization
    optimized_seq, final_score = optimizer.optimize(initial_seq, iterations=iterations)

    # Save the polished sequence, overwriting the draft or saving to a new file
    output_fasta = initial_fasta.replace(".fasta", "_polished.fasta")

    record = SeqRecord(
        Seq(optimized_seq),
        id=f"{seq_id}_MCMC_Polished|Score:{final_score:.4f}",
        description=""
    )

    with open(output_fasta, "w") as out:
        SeqIO.write(record, out, "fasta")

    print(f"INFO: Saved globally optimized and contextually polished sequence to {output_fasta}")