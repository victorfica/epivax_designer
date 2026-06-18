import numpy as np
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp


def solve_vaccine_atsp(epitopes, score_matrix):
    """
    Solves the optimal string-of-beads sequence using OR-Tools ATSP.

    :param epitopes: List of epitope string names.
    :param score_matrix: 2D numpy array of additive log-likelihood scores.
                         score_matrix[i][j] is the max score of linking E_i -> E_j
    :return: Optimal ordered list of epitopes.
    """
    num_epitopes = len(epitopes)

    # 1. Scale and Invert the Matrix (Maximize Score -> Minimize Cost)
    # Find the maximum score in the matrix to use as a baseline for inversion
    max_score = np.max(score_matrix)

    # We add 1 row and 1 column for the Dummy Node (Index 0)
    # The dummy node acts as the start and end of the linear polypeptide chain
    cost_matrix = np.zeros((num_epitopes + 1, num_epitopes + 1), dtype=int)

    for i in range(num_epitopes):
        for j in range(num_epitopes):
            if i == j:
                cost_matrix[i + 1][j + 1] = 0  # Cannot link an epitope to itself
            else:
                # Invert: Higher score becomes lower cost.
                # Multiply by 1000 to preserve precision as an integer.
                cost = int((max_score - score_matrix[i][j]) * 1000)
                cost_matrix[i + 1][j + 1] = cost

    # Dummy node distances to/from all epitopes remain 0 (already set by np.zeros)

    # 2. Initialize the OR-Tools Routing Index Manager
    manager = pywrapcp.RoutingIndexManager(len(cost_matrix), 1, 0)
    routing = pywrapcp.RoutingModel(manager)

    # 3. Create the Distance Callback
    def distance_callback(from_index, to_index):
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return cost_matrix[from_node][to_node]

    # set the cost of travel
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

    # 4. Set Search Parameters
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    # Use PATH_CHEAPEST_ARC for the initial heuristic
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)
    # Use GUIDED_LOCAL_SEARCH to escape local minima and find global optimum
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH)
    search_parameters.time_limit.seconds = 30 # 30 sec to optimize
    search_parameters.log_search = True

    # 5. Solve the Problem
    solution = routing.SolveWithParameters(search_parameters)

    # 6. Extract the Route
    if solution:
        index = routing.Start(0)
        route_indices = []
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0:  # Ignore the Dummy Node
                # Subtract 1 because epitopes list is 0-indexed, but matrix is 1-indexed
                route_indices.append(node_index - 1)
            index = solution.Value(routing.NextVar(index))

        optimal_order = [epitopes[i] for i in route_indices]

        # Calculate final sequence score (optional verification)
        total_score = 0
        for i in range(len(route_indices) - 1):
            total_score += score_matrix[route_indices[i]][route_indices[i + 1]]

        return optimal_order, total_score
    else:
        raise Exception("Solver failed to find a valid route.")


# ==========================================
# Practical Execution Example
# ==========================================
if __name__ == "__main__":
    # Let's say we have 3 epitopes
    my_epitopes = ["E1", "E2", "E3"]

    # Fake processing scores (log-likelihoods) generated from Phase 1
    # Rows = From Epitope, Cols = To Epitope
    # For example, E3 -> E1 has a massive score of 5.0
    # E1 -> E2 has a high score of 4.5
    fake_scores = np.array([
        # To E1,  To E2,  To E3
        [0.0, 4.5, -1.2],  # From E1
        [-2.0, 0.0, 1.5],  # From E2
        [5.0, 0.5, 0.0]  # From E3
    ])

    print("Running ATSP Graph Assembly...")
    optimal_path, final_score = solve_vaccine_atsp(my_epitopes, fake_scores)

    print(f"Optimal Polypeptide Chain: {' -> '.join(optimal_path)}")
    print(f"Total Framework Score: {final_score:.4f}")

    # Expected Output: E3 -> E1 -> E2 (Score: 5.0 + 4.5 = 9.5)