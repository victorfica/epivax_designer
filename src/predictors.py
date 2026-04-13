import torch
import pandas as pd
import lightning.pytorch as pl
from torch.utils.data import TensorDataset, DataLoader
import argparse

# Explicit imports relying on the installed mhcicleave package
from mhcicleave import config
from mhcicleave.models import MLP
from mhcicleave.lightning_model import CleavageLightningModule
from mhcicleave.cleavagedatasets import MultiFastaWindowDataset
from mhcicleave.encodings import Embedder


def run_cleavage_inference(
        fasta_file: str,
        output_file: str,
        checkpoint: str = None,
        emb_type: str = "concat",
        device_str: str = "cuda",
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
        emb_type (str): Type of embedding representation ('concat', 'mean', etc.).
        device_str (str): Target device ('cuda' or 'cpu').
        batch_size (int): Batch size for the MLP prediction.
        batch_size_emb (int): Batch size for ESM-2 embedding generation.
        num_workers (int): Number of CPU workers for dataloaders.
    """
    # 1. Setup Environment
    torch.set_float32_matmul_precision('medium')
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')

    if checkpoint is None:
        checkpoint = config.DEFAULT_CHECKPOINT
        print(f"INFO: No custom checkpoint provided. Using bundled model: {checkpoint}")
    else:
        print(f"INFO: Using custom user-provided checkpoint: {checkpoint}")

    print("--- MHC-I Cleavage Site Prediction ---")
    print(f"INFO: Using device: {device}")

    # 2. Initialize model using the package's internal config
    mlp_network = MLP(**config.DEFAULT_MLP_PARAMS)

    # 3. Load the trained Lightning Module
    print(f"INFO: Loading cleavage model from: {checkpoint}")
    try:
        model = CleavageLightningModule.load_from_checkpoint(checkpoint, model=mlp_network)
    except Exception as e:
        print(f"ERROR: Failed to load model checkpoint: {e}")
        return

    # 4. Prepare Data and Embeddings
    print(f"INFO: Loading sequence data from: {fasta_file}")
    window_dataset = MultiFastaWindowDataset(
        fasta_file=fasta_file,
        upstream=config.UPSTREAM,
        downstream=config.DOWNSTREAM
    )

    # IMPORTANT: Keep metadata before creating the embedding-only TensorDataset
    windowdata_df = pd.DataFrame(list(window_dataset))

    print("INFO: Initializing ESM-2 embedder...")
    embedder = Embedder(name=config.ESM2_MODEL_NAME, device=device)

    print("INFO: Generating contextual embeddings for all windows...")
    seq_embeddings = embedder.encode_esm(
        window_dataset,
        emb_type=emb_type,
        batch_size=batch_size_emb,
        num_workers=num_workers
    )

    # 5. Create DataLoader and Run Prediction
    embedding_dataset = TensorDataset(seq_embeddings)
    dataloader = DataLoader(
        embedding_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    print("INFO: Running MLP prediction batches...")
    trainer = pl.Trainer(accelerator='auto', devices=1, logger=False)
    prediction_batches = trainer.predict(model, dataloader)

    # 6. Process and Save Results
    print("INFO: Processing and saving cleavage probabilities...")
    predictions = torch.cat([batch.flatten() for batch in prediction_batches]).cpu().numpy()

    # Combine predictions with the saved metadata
    windowdata_df['prediction'] = predictions

    # Sort for a clean output file
    windowdata_df.sort_values(by=['prot_id', 'position'], inplace=True)
    windowdata_df.to_csv(output_file, index=False)

    print(f"\n✅ Predictions successfully saved to: {output_file}")


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

    args = parser.parse_args()

    # Unpack the argparse namespace into the new programmatic function
    run_cleavage_inference(
        fasta_file=args.fasta_file,
        checkpoint=args.checkpoint,
        output_file=args.output_file,
        emb_type=args.emb_type,
        device_str=args.device,
        batch_size=args.batch_size,
        batch_size_emb=args.batch_size_emb,
        num_workers=args.num_workers
    )