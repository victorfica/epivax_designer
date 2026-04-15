# EpivaxDesigner Workflow

epivax_designer/
│
├── vax_designer.py          # The main CLI entry point (handles all sub-commands)
├── requirements.txt         # Dependencies (pandas, biopython, torch, pytorch-lightning, mhcicleave)
│
├── src/                     # The Vaccine Assembly Logic
│   ├── __init__.py
│   ├── assembly.py          # Combinatorial pair seeding and linker generation
│   ├── predictor.py         # Wrapper calling mhcicleave for fasta predictions
│   ├── scoring.py           # Calculates protection/processing scores & plotting
│   ├── graph.py             # Translates doublets to a Hamiltonian path (Global Opt.)
│   └── optimizer.py         # MCMC logic for the final context refinement (Local Opt.)
│
├── data/                    # Directory for input files
│   └── input_epitopes.csv   # Example input: CSV with a 'peptide' column
│
└── results/                 # Directory for pipeline outputs (auto-generated)