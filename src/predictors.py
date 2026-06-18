import torch
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import argparse
from Bio import SeqIO
import gc
import os

from mhcicleave import config
from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
from mhcicleave.cleavagedatasets import MultiFastaWindowDataset
from mhcicleave.encodings import Embedder


def run_cleavage_inference(
        fasta_file: str,
        output_file: str,
        checkpoint: str = None,
        chunk_size: int = 2000,
        batch_size: int = 32,
        batch_size_emb: int = 32,
        num_workers: int = 4
):
    """
    Runs the MHC-I cleavage prediction pipeline programmatically.

    Args:
        fasta_file (str): Path to the input FASTA file containing constructs.
        output_file (str): Path to save the resulting predictions CSV.
        checkpoint (str): Path to the trained PyTorch Lightning model checkpoint.
        chunk_size (int): Number of sequences to process at a time.
        batch_size (int): Batch size for Dataloader during prediction.
        batch_size_emb (int): Batch size for ESM-2 embedding generation.
        num_workers (int): Number of CPU workers for dataloaders.
    """
    print("--- MHC-I Cleavage Site Prediction ---")

    # 1. Setup Environment
    torch.set_float32_matmul_precision('medium')
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    checkpoint = checkpoint if checkpoint else config.DEFAULT_CHECKPOINT
    mlp_network = MLP(**config.DEFAULT_MLP_PARAMS)

    try:
        model = CleavageLightningModule.load_from_checkpoint(checkpoint, model=mlp_network)
    except Exception as e:
        print(f"ERROR: Failed to load checkpoint: {e}")
        return

    print("INFO: Initializing ESM-2 embedder...")
    embedder = Embedder(name=config.ESM2_MODEL_NAME, device=device)

    # Read all records but only keep IDs and Sequences in RAM (very lightweight)
    all_records = list(SeqIO.parse(fasta_file, "fasta"))
    total_constructs = len(all_records)
    print(f"INFO: Found {total_constructs} constructs. Processing in chunks of {chunk_size}...")

    # Clear/Initialize the output CSV
    if os.path.exists(output_file):
        os.remove(output_file)

    first_chunk = True
    nchunk = 1

    for i in range(0, total_constructs, chunk_size):
        print(f"\n--- Processing Chunk {i // chunk_size + 1}/{(total_constructs // chunk_size) + 1} ---")

        chunk_records = all_records[i: i + chunk_size]
        temp_fasta = "temp_chunk.fasta"
        SeqIO.write(chunk_records, temp_fasta, "fasta")

        print(f"INFO: Loading sequence data for chunk {nchunk}")
        # Create dataset for just this chunk
        window_dataset = MultiFastaWindowDataset(
            fasta_file=temp_fasta,
            upstream=config.UPSTREAM,
            downstream=config.DOWNSTREAM
        )
        windowdata_df = pd.DataFrame(window_dataset.index, columns=['prot_id', 'position'])
        windowdata_df['window'] = [window_dataset[idx]['window'] for idx in range(len(window_dataset))]
        windowdata_df['position'] = windowdata_df['position'] + 1

        # Generate Embeddings (This is where the RAM spike used to happen)
        seq_embeddings = embedder.encode_esm(
            window_dataset,
            emb_type="concat",
            batch_size=batch_size_emb,
            num_workers=num_workers
        )

        print("INFO: Generating contextual embeddings for all windows...")
        # Predict
        embedding_dataset = TensorDataset(seq_embeddings)
        dataloader = DataLoader(embedding_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        trainer = pl.Trainer(accelerator='auto', devices=1, logger=False, enable_progress_bar=False)
        print("INFO: Running MLP prediction batches...")
        prediction_batches = trainer.predict(model, dataloader)
        print("INFO: Processing and saving cleavage probabilities...")
        predictions = torch.cat([batch.flatten() for batch in prediction_batches]).cpu().numpy()

        # Safely assign and sort
        if len(predictions) == len(windowdata_df):
            windowdata_df['prediction'] = predictions
            windowdata_df.sort_values(by=['prot_id', 'position'], inplace=True)
        else:
            print(f"ERROR: Shape mismatch in Chunk {nchunk}. Preds: {len(predictions)}, DF: {len(windowdata_df)}")
            break
        # Append mode for chunks after the first one
        mode = 'w' if first_chunk else 'a'
        windowdata_df.to_csv(output_file, mode=mode, header=first_chunk, index=False)

        first_chunk = False
        nchunk += 1

        # 3. GARBAGE COLLECTION (The crucial step to save your RAM)
        del window_dataset
        del windowdata_df
        del seq_embeddings
        del embedding_dataset
        del dataloader
        del prediction_batches
        del predictions
        gc.collect()
        torch.cuda.empty_cache()

        # Clean up temp file
    if os.path.exists("temp_chunk.fasta"):
        os.remove("temp_chunk.fasta")

    print(f"\n✅ All chunks processed. Predictions saved to: {output_file}")



# --- STANDALONE CLI FALLBACK ---
# This allows you to still run `python src/predictor.py --fasta_file ...` directly if needed for debugging.
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run Cleavage Prediction Inference.")

    parser.add_argument("--fasta_file", type=str, required=True, help="Path to the input FASTA file.")
    parser.add_argument("--output_file", type=str, required=True, help="Path to save output predictions CSV.")

    parser.add_argument("--checkpoint", type=str, default="", help="Path to trained model .ckpt file.")
    parser.add_argument("--emb_type", type=str, default="concat", choices=["mean", "max", "concat"])
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--batch_size_emb", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--chunk_size", type=int, default=2000,
                             help="Number of sequences to process at once to prevent RAM OOM")

    args = parser.parse_args()

    # Unpack the argparse namespace into the new programmatic function
    run_cleavage_inference(
        fasta_file=args.fasta_file,
        checkpoint=args.checkpoint,
        chunk_size=args.chunk_size,
        output_file=args.output_file,
        batch_size=args.batch_size,
        batch_size_emb=args.batch_size_emb,
        num_workers=args.num_workers
    )