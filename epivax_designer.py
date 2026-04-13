#!/usr/bin/env python3
import argparse
import os
from pathlib import Path

from src.assemble import build_multifasta
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
    parser_asm = subparsers.add_parser("assemble", help="Generate combinatorial linkers")
    parser_asm.add_argument("--epitopes_csv", required=True, help="Input CSV with selected epitopes")
    parser_asm.add_argument("--output_fasta", required=True, help="Output constructs FASTA")
    parser_asm.add_argument("--num_linkers", type=int, default=5000)
    parser_asm.add_argument("--linker_len", type=int, default=4)
    parser_asm.add_argument("--structure", default="LELE", help="e.g., LELE, ELE")

    # --- 2. PREDICT ---
    parser_pred = subparsers.add_parser("predict", help="Run DL cleavage prediction")
    parser_pred.add_argument("--fasta_constructs", required=True, help="Constructs FASTA")
    parser_pred.add_argument("--checkpoint", default=None, help="Path to custom .ckpt. If omitted, uses default bundled model.")
    parser_pred.add_argument("--output_csv", required=True, help="Predictions output CSV")

    # --- 3. SCORE ---
    parser_score = subparsers.add_parser("score", help="Score and rank constructs")
    parser_score.add_argument("--predictions_csv", required=True)
    parser_score.add_argument("--fasta_constructs", required=True)
    parser_score.add_argument("--output_csv", required=True, help="Ranked output CSV")

    # --- 4. GRAPH ASSEMBLE ---
    parser_graph = subparsers.add_parser("graph", help="Stitch optimal doublets into a full vaccine chain")
    parser_graph.add_argument("--epitopes_csv", required=True, help="Original input epitopes")
    parser_graph.add_argument("--scored_doublets_csv", required=True, help="CSV output from the 'score' step")
    parser_graph.add_argument("--output_fasta", required=True, help="Output FASTA for the draft vaccine")

    # --- 5. OPTIMIZE (MCMC) ---
    parser_opt = subparsers.add_parser("optimize", help="Run MCMC linkers refinement")
    parser_opt.add_argument("--initial_fasta", required=True, help="Starting sequence(s)")
    parser_opt.add_argument("--checkpoint", required=True, help="Model .ckpt path")
    parser_opt.add_argument("--iterations", type=int, default=1000, help="MCMC steps")

    # --- 6. END-TO-END PIPELINE ---
    parser_pipe = subparsers.add_parser("pipeline", help="Run the full automated workflow")
    parser_pipe.add_argument("--epitopes_csv", required=True, help="Input CSV with selected epitopes")
    parser_pipe.add_argument("--checkpoint", default=None, help="Path to custom .ckpt. If omitted, uses default bundled model.")
    parser_pipe.add_argument("--num_linkers", type=int, default=5000, help="Linkers per doublet")
    parser_pipe.add_argument("--outdir", default="results", help="Directory to save pipeline files")
    parser_pipe.add_argument("--run_name", default="vax_run", help="Prefix for output files")
    parser_pipe.add_argument("--optimize", action="store_true", help="Include MCMC refinement at the end")

    args = parser.parse_args()

    # --- Routing ---
    if args.command == "pipeline":
        os.makedirs(args.outdir, exist_ok=True)
        prefix = f"{args.outdir}/{args.run_name}"

        print(f"\n=== STARTING PIPELINE: {args.run_name} ===")
        print(f"INFO: Using input epitopes from {args.epitopes_csv}")

        print("\n=== STEP 1: ASSEMBLE DOUBLETS ===")
        doublets_fasta = f"{prefix}_doublets.fasta"
        build_multifasta(args.epitopes_csv, args.num_linkers, 4, "ELE", doublets_fasta)

        print("\n=== STEP 2: PREDICT CLEAVAGE ===")
        predictions_csv = f"{prefix}_predictions.csv"
        run_cleavage_inference(doublets_fasta, args.checkpoint, predictions_csv)

        print("\n=== STEP 3: SCORE CONSTRUCTS ===")
        scored_csv = f"{prefix}_scored_doublets.csv"
        score_constructs(predictions_csv, doublets_fasta, scored_csv)

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
        build_multifasta(args.epitopes_csv, args.num_linkers, args.linker_len, args.structure, args.output_fasta)
    elif args.command == "predict":
        run_cleavage_inference(args.fasta_constructs, args.checkpoint, args.output_csv)
    elif args.command == "score":
        score_constructs(args.predictions_csv, args.fasta_constructs, args.output_csv)
    elif args.command == "graph-assemble":
        assemble_graph(args.epitopes_csv, args.scored_doublets_csv, args.output_fasta)
    elif args.command == "optimize":
        run_mcmc_optimization(args.initial_fasta, args.checkpoint, args.iterations)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()