#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from src.assemble import build_seeding_fasta
from src.predictors import run_cleavage_inference
from src.scoring import score_constructs
from src.graph import assemble_graph
from src.optimizer import run_mcmc_optimization


def main():
    parser = argparse.ArgumentParser(
        description="Rational Assembly of Multi-Epitope Vaccines Framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    subparsers = parser.add_subparsers(dest="command", help="Pipeline execution steps")

    # --- 1. ASSEMBLE (Combinatorial) ---
    parser_asm = subparsers.add_parser("assemble", help="Generate doublet combinatorial")
    parser_asm.add_argument("-i","--epitopes_csv", required=True, help="Input CSV with selected epitopes")
    parser_asm.add_argument("-o","--output_constructs", required=True, help="Output FASTA constructs")
    parser_asm.add_argument("-n","--num_linkers", type=int, default=5000, help="Number of random linkers to generate PER length")
    parser_asm.add_argument("-l", "--linker_lens", nargs="+", type=int, default=[4], choices=[2,3,4,5,6],
                             help="Linker lengths to generate (e.g., -l 3 -l 4 -l 5 or -l 3 4 5). Can specify multiple lengths.")
    #parser_asm.add_argument("-s","--structure", default="LELE", help="e.g., LELE, ELE")
    parser_asm.add_argument("-m", "--mode", type=str, choices=["doublet", "singlet"], default="doublet",
                            help="'doublet' for multi-epitope seeding, 'singlet' for single L-E extensions")

    # --- 2. PREDICT ---
    parser_pred = subparsers.add_parser("predict", help="Run cleavage predictions over constructs")
    parser_pred.add_argument("-i","--output_constructs", required=True, help="Input FASTA constructs")
    parser_pred.add_argument("-o","--output_predictions", required=True, help="Output predictions CSV")
    parser_pred.add_argument("--checkpoint", default=None, help="Path to custom .ckpt. If omitted, uses default bundled model.")
    parser_pred.add_argument("--chunk_size", type=int, default=5000,
                             help="Number of sequences to process at once to prevent RAM OOM")

    # --- 3. SCORE ---
    parser_score = subparsers.add_parser("score", help="Score and rank constructs")
    parser_score.add_argument("-i", "--output_predictions", required=True, help="Input predictions CSV")
    parser_score.add_argument("--output_constructs", required=True, help="Input FASTA constructs")
    parser_score.add_argument("-o", "--score_csv", required=True, help="Output score CSV")

    # --- 4. GRAPH ASSEMBLE ---
    parser_graph = subparsers.add_parser("graph", help="Stitch optimal doublets into a full vaccine chain")
    parser_graph.add_argument("--epitopes_csv", required=True, help="Original input epitopes")
    parser_graph.add_argument("--score_csv", required=True, help="CSV output from the 'score' step")
    parser_graph.add_argument("--output_draft", required=True, help="Output FASTA for the draft vaccine")

    # --- 5. OPTIMIZE (bGA) ---
    parser_opt = subparsers.add_parser("optimize", help="Run MCMC linkers refinement")
    parser_opt.add_argument("-i", "--output_draft", required=True, help="Starting draft sequence(s)")
    parser_opt.add_argument("--checkpoint", required=True, help="Cleavage model .ckpt path")
    parser_opt.add_argument("--generations", type=int, default=50, help="Number of evolutionary epochs")
    parser_opt.add_argument("--pop_size", type=int, default=64, help="GPU batch evaluation size")

    # --- 6. END-TO-END PIPELINE ---
    parser_pipe = subparsers.add_parser("pipeline", help="Run the full automated workflow")
    parser_pipe.add_argument("-i", "--epitopes_csv", required=True, help="Input CSV with selected epitopes")
    parser_pipe.add_argument("-n", "--num_linkers", type=int, default=5000,
                            help="Number of random linkers to generate PER length")
    parser_pipe.add_argument("-l", "--linker_lens", nargs="+", type=int, default=[4], choices=[2,3,4,5,6],
                            help="Linker lengths to generate (e.g., -l 3 -l 4 -l 5 or -l 3 4 5). Can specify multiple lengths.")
    parser_pipe.add_argument("-m", "--mode", type=str, choices=["doublet", "singlet"], default="doublet",
                            help="'doublet' for multi-epitope seeding, 'singlet' for single L-E extensions")

    parser_pipe.add_argument("--checkpoint", default=None, help="Path to custom .ckpt. If omitted, uses default bundled model.")

    parser_pipe.add_argument("--outdir", default="results", help="Directory to save pipeline files")
    parser_pipe.add_argument("--run_name", default="vax_run", help="Prefix for output files")
    parser_pipe.add_argument("--optimize", action="store_true", help="Include MCMC refinement at the end")
    parser_pipe.add_argument("--chunk_size", type=int, default=2000,
                             help="Number of sequences to process at once to prevent RAM OOM")

    args = parser.parse_args()

    # --- Routing ---
    if args.command == "pipeline":
        os.makedirs(args.outdir, exist_ok=True)
        prefix = f"{args.outdir}/{args.run_name}"

        print(f"\n=== STARTING PIPELINE: {args.run_name} ===")
        print(f"INFO: Using input epitopes from {args.epitopes_csv}")

        print("\n=== STEP 1: ASSEMBLE DOUBLETS ===")
        output_fasta = f"{prefix}_doublets.fasta"
        build_seeding_fasta(args.epitopes_csv, args.num_linkers, args.linker_lens, args.mode, output_fasta)

        print("\n=== STEP 2: PREDICT CLEAVAGE ===")
        predictions_csv = f"{prefix}_predictions.csv"
        run_cleavage_inference(args.fasta_constructs, predictions_csv, args.checkpoint, args.chunk_size)

        print("\n=== STEP 3: SCORE CONSTRUCTS ===")
        scored_csv = f"{prefix}_scored_doublets.csv"
        score_constructs(predictions_csv, output_fasta, scored_csv)

        print("\n=== STEP 4: GRAPH ASSEMBLY (Global Optimum) ===")
        draft_fasta = f"{prefix}_draft_optimal.fasta"
        assemble_graph(args.epitopes_csv, scored_csv, draft_fasta)

        if args.optimize:
            print("\n=== STEP 5: MCMC OPTIMIZATION (Local Context Polish) ===")
            run_mcmc_optimization(draft_fasta, args.checkpoint, iterations=1000)
        else:
            print("\n=== PIPELINE COMPLETE (No MCMC) ===")
            print(f"Top scored sequence saved to: {draft_fasta}")

    elif args.command == "assemble":
        print(args.linker_lens)
        build_seeding_fasta(args.epitopes_csv, args.num_linkers, args.linker_lens, args.mode, args.output_constructs)
    elif args.command == "predict":
        run_cleavage_inference(args.output_constructs, args.output_predictions, args.checkpoint, args.chunk_size)
    elif args.command == "score":
        score_constructs(args.output_predictions, args.output_constructs, args.score_csv)
    elif args.command == "graph":
        assemble_graph(args.epitopes_csv, args.score_csv, args.output_fasta)
    elif args.command == "optimize":
        run_mcmc_optimization(args.initial_fasta, args.checkpoint, args.iterations)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()