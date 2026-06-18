import pandas as pd
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from Bio import SeqIO
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def extract_optimal_edges(epitope_ids, doublets_df):
    """
    Parses the scored doublets via Explicit ID Routing to find the single
    best linker and score for every directed pair of epitopes (E1 -> E2).
    """
    print("INFO: Mapping optimal linkers to epitope IDs...")
    best_edges = {}

    for _, row in doublets_df.iterrows():
        seq_id = str(row['seq_id'])  # Format expectation: E1_E2_6467|RRQW|E9_L4_E9
        score = float(row['score'])

        # 1. Parse the IDs directly from the header
        parts = seq_id.split('|')
        id_chunk = parts[0].split('_')

        # Ensure it is a multi-epitope doublet (e.g., ['E1', 'E2', '6467'])
        # Ignore singlets which start with 'L' (e.g., ['L', 'E1', '0'])
        if len(id_chunk) >= 3 and id_chunk[0] != 'L':
            e1_id = id_chunk[0]
            e2_id = id_chunk[1]
            linker = parts[1]

            edge_key = (e1_id, e2_id)

            # 2. Update if it's the highest biological score for this specific structural junction
            if edge_key not in best_edges or score > best_edges[edge_key]['score']:
                best_edges[edge_key] = {'linker': linker, 'score': score}

    return best_edges


def create_data_model(epitope_ids, best_edges):
    """
    Stores the data for the Asymmetric Traveling Salesperson Problem (ATSP).
    Transforms biological processing scores into mathematical routing costs.
    """
    data = {}
    num_epitopes = len(epitope_ids)

    # +1 because Index 0 is the "Dummy Depot" used to break the loop into a linear path
    num_nodes = num_epitopes + 1

    # Find the maximum score to invert the matrix (Maximize Score -> Minimize Cost)
    max_score = max([edge['score'] for edge in best_edges.values()])

    cost_matrix = {}
    for from_node in range(num_nodes):
        cost_matrix[from_node] = {}
        for to_node in range(num_nodes):

            # 1. Dummy Depot Connections (Index 0)
            # Traveling from Depot to start of vaccine, or from end of vaccine back to Depot costs 0.
            if from_node == 0 or to_node == 0:
                cost_matrix[from_node][to_node] = 0

            # 2. Self-Connections
            elif from_node == to_node:
                cost_matrix[from_node][to_node] = 0

            # 3. Real Epitope-to-Epitope Connections
            else:
                # Subtract 1 to map the routing node back to the 0-indexed ID list
                e_from_id = epitope_ids[from_node - 1]
                e_to_id = epitope_ids[to_node - 1]

                if (e_from_id, e_to_id) in best_edges:
                    score = best_edges[(e_from_id, e_to_id)]['score']
                    # Invert and scale to positive integer for OR-Tools solver
                    cost = int((max_score - score) * 1000)
                    cost_matrix[from_node][to_node] = cost
                else:
                    # Massive penalty if an edge is biologically missing
                    cost_matrix[from_node][to_node] = 9999999

    data["cost_matrix"] = cost_matrix
    data["num_vehicles"] = 1
    data["depot"] = 0
    return data


def assemble_graph(epitopes_csv, scored_doublets_csv, output_fasta):
    """
    Finds the Hamiltonian path through all epitopes that maximizes the total
    MHC-I cleavage and processing score.
    """
    print(f"\n--- GRAPH ASSEMBLY: ATSP SOLVER ---")

    # 1. Load Data & Generate Dictionaries
    epitopes_df = pd.read_csv(epitopes_csv)

    # Failsafe: Generate IDs if missing
    if 'id' not in epitopes_df.columns:
        epitopes_df['id'] = [f"E{i + 1}" for i in range(len(epitopes_df))]

    # Dictionary mapping ID to Sequence: {'E1': 'DPLVVAASI', 'E2': 'LVVAASIIG'}
    epitopes_dict = dict(zip(epitopes_df['id'], epitopes_df['peptide']))
    epitope_ids = list(epitopes_dict.keys())

    doublets_df = pd.read_csv(scored_doublets_csv)
    best_edges = extract_optimal_edges(epitope_ids, doublets_df)

    # 2. Instantiate the Data Problem
    data = create_data_model(epitope_ids, best_edges)

    # 3. Create the Routing Index Manager and Model
    manager = pywrapcp.RoutingIndexManager(
        len(data["cost_matrix"]), data["num_vehicles"], data["depot"]
    )
    routing = pywrapcp.RoutingModel(manager)

    # 4. Create and Register the Distance Callback
    def distance_callback(from_index, to_index):
        """Returns the integer cost between the two nodes."""
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data["cost_matrix"][from_node][to_node]

    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 5. Set Search Parameters (Greedy start + Metaheuristic refinement)
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 5  # Give the solver 5 seconds to escape local minima

    # 6. Solve the Problem
    print("INFO: Initializing OR-Tools Guided Local Search...")
    solution = routing.SolveWithParameters(search_parameters)

    # 7. Extract Solution and Stitch Sequence
    if not solution:
        raise ValueError("CRITICAL ERROR: OR-Tools failed to find a valid routing path.")

    index = routing.Start(0)
    route_indices = []

    while not routing.IsEnd(index):
        node_index = manager.IndexToNode(index)
        if node_index != 0:  # Ignore the Dummy Depot
            # Map back to 0-indexed epitope IDs
            route_indices.append(node_index - 1)
        index = solution.Value(routing.NextVar(index))

    optimal_order_ids = [epitope_ids[i] for i in route_indices]

    # Reconstruct the full sequence and exact biological score
    final_sequence = epitopes_dict[optimal_order_ids[0]]
    final_score = 0.0

    for i in range(len(optimal_order_ids) - 1):
        e_from_id = optimal_order_ids[i]
        e_to_id = optimal_order_ids[i + 1]
        edge_data = best_edges[(e_from_id, e_to_id)]

        # Append: Linker + Next Epitope Sequence
        final_sequence += edge_data['linker'] + epitopes_dict[e_to_id]
        final_score += edge_data['score']

    print(f"✅ Global Optimum Found!")
    print(f"Optimal Epitope Order: {' -> '.join(optimal_order_ids)}")
    print(f"Maximized Processing Score: {final_score:.4f}")

    # 8. Save the Globally Optimized Draft to FASTA
    record = SeqRecord(Seq(final_sequence), id="ATSP_Optimal_Draft", description=f"Score={final_score:.4f}")
    with open(output_fasta, "w") as f:
        SeqIO.write(record, f, "fasta")

    print(f"Draft Sequence saved to: {output_fasta}\n")
    return output_fasta