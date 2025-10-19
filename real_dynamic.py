from random import random, randint, sample, choices, seed, random, Random
from copy import deepcopy
import math, joblib, heapq, tracemalloc
from time import time
from multiprocessing import Pool
from map_Generator import plot_paths
from graph_generator import generateGraph

def floydWarshall(graph:list, max_iter:int):
    """
    Implement Floyd-Warshall Algorithm to find the shortest path
    """
    nodes = len(graph)
    distance = deepcopy(graph)
    dynamic_change_list = [max_iter]*len(graph)
    same_random_gen = Random(100)
    change_dict = dict()
    
    next_node = []
    for i in range(nodes):
        next_node.append([None]*nodes)
    for start in range(nodes):
        for end in range(nodes):
            if distance[start][end] != -1:
                next_node[start][end] = end  # if there's a direct edge from i to j
    counter = 0
    for mid in range(nodes):
        for start in range(nodes):
            for end in range(nodes):
                dynamic_change(dynamic_change_list, same_random_gen, graph, change_dict) # change the weight of edges connected to a specific node
                if (distance[start][end] == -1 or distance[start][end] > (distance[start][mid] + distance[mid][end])) \
                and (distance[start][mid] != -1 and distance[mid][end] != -1):
                    distance[start][end] = distance[start][mid] + distance[mid][end]
                    next_node[start][end] = next_node[start][mid]
                counter+=1

    return distance, next_node, graph

def reconstruct_path(start:int, end:int, next_node:list, original_graph:list):
    """
    reconstruct the path using the nextNode table returned by floydWarshall function
    """
    if next_node[start][end] is None:
        return None, -1  # no path
    path = [start]
    wei = 0
    while start != end:
        mid = next_node[start][end]
        if mid is None:
            print("invalid path")
            return [], 0
        # print(start, mid, end)
        wei+=original_graph[start][mid]
        start = mid
        path.append(start)
    return path, wei

def generateAdjacencyMatrix(nodes:int, max_weigth:int, density:float):
    """
    Generate a random adjacency matrix with specified density
    """
    graph = []
    for i in range(nodes):
        graph.append([-1]*nodes)
    for i in range(nodes):
        #makes diags 0
        graph[i][i]=0
    for i in range(nodes):
        for j in range(nodes):
            #different values to go/return to same location. ex: it could take a weight of 1 to go from a to b, but a weight of inf to go from b to a.
            if  i != j and random() < density:
                graph[i][j] = randint(1, max_weigth)
    return graph

def getCost(solution:list, graph:list, temp:float, maxWei=2000):
    """
    Calculate the cost of the path
    """
    base_penalty = maxWei*2e4  # this can be tuned 
    penalty = int(round(base_penalty / (temp**0.5 + 1)))
    # print(penalty)
    cost = 0
    for i in range(len(solution) - 1):
        a, b = solution[i], solution[i + 1]
        wei = graph[a][b]
        if wei == -1:
            cost += penalty
        else:
            cost += wei
    return cost

def swap2Nodes(solution:list):
    if len(solution)<4: return -1        
    node1, node2 = sample(solution[1:-1], 2)
    index1, index2 = solution.index(node1), solution.index(node2)
    
    temp = solution.copy()
    temp[index1] = node2
    temp[index2] = node1

    return temp

def swap_subroute(solution:list):
    if len(solution)<5: return -1   
    fail_counter = 0
    for i in range(100):  
        index1, index2 = sorted(sample(range(1, len(solution)-1), 2))
        subroute = solution[index1:index2+1]
        rest = solution[:index1] + solution[index2+1:]
        if len(rest) < 3:
            fail_counter+=1
            if fail_counter==10: break
            continue  # not enough to insert between two nodes
        available = list(range(1, len(rest)))
        available.remove(index1)
        insert_pos = sample(available, 1)[0]
        temp = rest[:insert_pos] + subroute + rest[insert_pos:]
        
        return temp
    
    return -1

def inverse(solution:list):
    if len(solution)<4: return -1        
    index1, index2 = sorted(sample(range(1,len(solution)-1), 2))
    
    parts = solution[index1:index2+1]
    parts.reverse()
    temp = solution[:index1] + parts + solution[index2+1:]
    
    return temp

def insert(solution:list, graph:list):
    available = []
    for i in range(len(graph)):
        if i in solution: continue
        available.append(i)
        
    if len(available)==0: return -1
    
    node = sample(available, 1)[0]
    index = randint(1, len(solution)-1)
    
    temp = solution.copy()
    temp.insert(index, node)

    return temp

def remove(solution:list):
    if len(solution)<3: return -1
    index = randint(1, len(solution)-2)
    temp = solution.copy()
    del temp[index]
    
    return temp

def replace_connected_to(solution:list, graph:list):
    if len(solution)<3: return -1
    index_list = list(range(0, len(solution)-2))
    while len(index_list)!=0:
        start_index = sample(index_list, 1)[0]
        
        ban_list = set(solution[:start_index+2])
        ban_list.add(solution[-1])
        
        start = solution[start_index]
        connection = graph[start]
        end_list = []
        
        for end, end_wei in enumerate(connection):
            if end not in ban_list and end_wei>0: 
                end_list.append(end)
                
        if len(end_list)!=0: 
            end = sample(end_list, 1)[0]
            temp = solution.copy()
            temp[start_index+1] = end
            return temp
        else: index_list.remove(start_index)
        
    return -1

def replace_connected_from(solution:list, graph:list):
    if len(solution)<3: return -1
    index_list = list(range(2, len(solution)))
    while len(index_list)!=0:
        start_index = sample(index_list, 1)[0]
        
        ban_list = set(solution[start_index-1:])
        ban_list.add(solution[0])
        
        start = solution[start_index]
        connection = graph[start]
        end_list = []

        for end, end_wei in enumerate(connection):
            if end not in ban_list and end_wei>0: 
                end_list.append(end)
            
        if len(end_list)!=0: 
            end = sample(end_list, 1)[0]
            temp = solution.copy()
            temp[start_index-1] = end
            return temp
        else: index_list.remove(start_index)
        
    return -1

def insert_connected_to(solution:list, graph:list):
    insert_index = randint(0, len(solution)-2)
    node = solution[insert_index]
    connection = graph[node]
    connect_list = []
    for connect, wei in enumerate(connection):
        if wei<1 or connect in solution: continue
        connect_list.append(connect)
        
    if len(connect_list)==0: return -1
    
    temp = solution.copy()
    temp.insert(insert_index+1, sample(connect_list, 1)[0])

    return temp

def insert_connected_from(solution:list, graph:list):
    insert_index = randint(1, len(solution)-1)
    node = solution[insert_index]
    connection = graph[node]
    connect_list = []
    for connect, wei in enumerate(connection):
        if wei<1 or connect in solution: continue
        connect_list.append(connect)
        
    if len(connect_list)==0: return -1

    temp = solution.copy()
    temp.insert(insert_index, sample(connect_list, 1)[0])

    return temp

def getNeighbors(solution: list, graph: list, temperature: float, t_max:float, success_list:list, attempt_list:list, preferences:list):
    """
    Generate neighbor solution based on current solution
    """
    # Define weights based on temperature (normalized between 0 and 1)
    t_norm = min(1.0, temperature / t_max*1.0)  
    wei_list = [
        # base wei + additional preference
        # By using nonlinear functions, delay or amplify mutation behaviors at different stages.
        0.1 + 0.1 * t_norm**2,          # swap2Nodes, Medium disruption
        0.1 + 0.1 * t_norm**2,          # swap_subroute, More aggressive
        0.1 + 0.1 * t_norm**2,          # inverse
        0.2 + 0.2 * t_norm**2,          # insert random, Necessary at the begin to explore
        0.2 + 0.3 * (1 - t_norm**2),    # remove, More likely at low temp
        0.2 + 0.2 * t_norm**2,          # replace, Necessary at the begin to explore
        0.2 + 0.2 * t_norm**2,          # replace, Necessary at the begin to explore
        0.2 + 0.2 * t_norm**2,          # insert connect to, Necessary at the begin to explore
        0.2 + 0.2 * t_norm**2           # insert connect from, Necessary at the begin to explore
    ]
    
    # apply preferences
    prefer_wei_list = []
    for wei, pref in zip(wei_list, preferences):
        prefer_wei_list.append(wei*pref)
        
    # apply success tracking to favor effective mutation
    adaptive_wei_list = []
    for suc, att, wei in zip(success_list, attempt_list, prefer_wei_list):
        adaptive_wei_list.append((suc/att)*wei)
        
    # Normalize weights
    total = sum(adaptive_wei_list)
    probs = []
    for wei in adaptive_wei_list:
        probs.append(wei/total)

    option_list = list(range(9))
    neighbor = -1
    while neighbor==-1 and len(option_list)!=0:
        number = sample(option_list, 1)[0]
        number = choices(option_list, weights=probs, k=1)[0]
        if number==0: 
            neighbor = swap2Nodes(solution) # only valid when the solution len>3
            if neighbor==-1: 
                index = option_list.index(0)
                del option_list[index]
                del probs[index]
        elif number==1: 
            neighbor = swap_subroute(solution)# only valid when the solution len>4
            if neighbor==-1: 
                index = option_list.index(1)
                del option_list[index]
                del probs[index]
        elif number==2: 
            neighbor = inverse(solution)# only valid when the solution len>3
            if neighbor==-1: 
                index = option_list.index(2)
                del option_list[index]
                del probs[index]
        elif number==3: 
            neighbor = insert(solution, graph)# work when the solution does not use all nodes
            if neighbor==-1: 
                index = option_list.index(3)
                del option_list[index]
                del probs[index]
        elif number==4: 
            neighbor = remove(solution)# only valid when the solution len>2
            if neighbor==-1: 
                index = option_list.index(4)
                del option_list[index]
                del probs[index]
        elif number==5:
            neighbor = replace_connected_to(solution, graph)# only valid when the solution len>2
            if neighbor==-1: 
                index = option_list.index(5)
                del option_list[index]
                del probs[index]
        elif number==6:
            neighbor = replace_connected_from(solution, graph)# only valid when the solution len>2
            if neighbor==-1: 
                index = option_list.index(6)
                del option_list[index]
                del probs[index]
        elif number==7:
            neighbor = insert_connected_to(solution, graph)
            if neighbor==-1: 
                index = option_list.index(7)
                del option_list[index]
                del probs[index]
        elif number==8:
            neighbor = insert_connected_from(solution, graph)
            if neighbor==-1: 
                index = option_list.index(8)
                del option_list[index]
                del probs[index]
    if neighbor!=-1: 
        attempt_list[number]+=1
        return neighbor, number
    else: return -1, -1 # cannot modify the current solution
    
def estimate_initial_temp(solution:list, graph:list, pref:list, trials=100, desired_acceptance=0.8):
    # We want to simulate how bad neighbors are early on, so we should use a high "fake" temperature when estimating.
    fake_temp=2000
    total_delta = 0
    more_expensive = 0
    current_cost = getCost(solution, graph, fake_temp)
    temp_mutation_success = [1] * 9
    temp_mutation_attempts = [1] * 9
    
    for _ in range(trials):
        neighbor, mutation_index = getNeighbors(solution, graph, fake_temp, fake_temp, temp_mutation_success, temp_mutation_attempts, pref)
        if neighbor == -1: continue  # invalid move
        delta = getCost(neighbor, graph, fake_temp) - current_cost
        if delta > 0: # new solution is more expensive
            total_delta += delta
            more_expensive += 1
        else:
            temp_mutation_success[mutation_index]+=1

    if more_expensive == 0:
        return 1  # fallback
    avg_delta = total_delta / more_expensive
    return -avg_delta / math.log(desired_acceptance)

def get_max_iterations(num_nodes:int, base_max_iter=100_000):
    scale = (math.log(num_nodes + 1) / math.log(100 + 1))  # Normalize against 100 nodes
    return max(base_max_iter, int(base_max_iter * scale))

def dynamic_change(change_list:list, random_gen:Random, graph:list, change_dict:dict, decrease_factor = 0.9):    
    """
    Change the weight of edges of a random node in the adjacency graph
    """
    # traffic_condition_factors = {
    # # Negative effects (increase cost)
    # "high_traffic":      1.008,
    # "medium_traffic":    1.004,
    # "accident":          1.005,
    # "bad_weather":       1.002,
    # "construction":      1.006,
    # "special_event":     1.005,
    # "road_closure":      1.02,  # Can be treated as blocked

    # # Positive effects (decrease cost)
    # "green_wave":        0.975,
    # "off_peak_hours":    0.99,
    # "priority_road":     0.97,
    # }
    
    traffic_factor = [1.008, 1.004, 1.005, 1.002, 1.006, 1.005, 1.02, 0.975, 0.99, 0.97]
    change_node = random_gen.choices(range(len(change_list)), weights=change_list)[0]
    connection = graph[change_node]
    traffic = random_gen.choices(traffic_factor)[0]
    for i, wei in enumerate(connection):
            if wei>0 and random_gen.random()>0.8: connection[i] = max(int(round(wei*traffic)),1)
   
def simulated_annealing(solution:list, graph:list, pref:list, max_iter:int, random_seed:int, T_min=1e-4):
    """
    Using Simulated Annealing to find the shortest path
    """
    seed(random_seed)
    T = estimate_initial_temp(solution, graph, pref)
    T0 = T
    current = solution
    start_solution = current.copy()
    current_cost = getCost(current, graph, T)
    best = current
    best_cost = current_cost
    stagnation_counter = 0
    no_improvement_for = 0
    mutation_success = [1] * 9
    mutation_attempts = [1] * 9
    # dynamic environment
    dynamic_change_list = [max_iter]*len(graph)
    same_random_gen = Random(100)
    change_dict = dict()
    
    # visualization
    path_list = [current]
    
    best_index = max_iter
    
    
    for i in range(1, max_iter+1):
        if current!=path_list[-1]: path_list.append(current)
        dynamic_change(dynamic_change_list, same_random_gen, graph, change_dict) # change the weight of edges connected to a specific node
            
        if T < T_min:
            break

        neighbor, mutation_index = getNeighbors(current, graph, T, T0, mutation_success, mutation_attempts, pref)
        if neighbor == -1:
            print("stuck")
            T *= 1.2  # reheat to escape being stuck
            current = best
            continue
        
        neighbor_cost = getCost(neighbor, graph, T)
        current_cost = getCost(current, graph, T) # recalculate the current cost bc the graph may be changed
        delta = neighbor_cost - current_cost

        # Adaptive cooling: cool slow when good solution, fast when bad solution
        if delta < 0 or random() < math.exp(-delta / T):
            mutation_success[mutation_index]+=1
            current = neighbor
            current_cost = neighbor_cost
            best_cost = getCost(best, graph, T) # recalculate the best cost bc the graph may be changed
            if current_cost < best_cost:
                best = current
                best_cost = current_cost
                best_index = i
                
            T *= 0.99995
            stagnation_counter = 0
            no_improvement_for = 0
        else:
            stagnation_counter += 1
            no_improvement_for += 1
            if stagnation_counter > 50:
                T *= 0.99
                stagnation_counter = 0
            if no_improvement_for > 150:
                T *= 1.2  # reheat
                no_improvement_for = 0           
    
    start_cost = getCost(start_solution, graph, T)
    return best, best_cost, start_solution, start_cost, graph, path_list, 1 - best_index/max_iter

def sa_runner(args):
    start_solution, graph, seed_value, nodes, pref = args
    return simulated_annealing(start_solution, graph, pref, get_max_iterations(nodes), seed_value)

def run_parallel_sa(graph:list, nodes:int, start_solution_path:list, amount:int):
    # Preference multipliers per solver
    # swapNode, swapRoute, inverse, insert, remove, replaceTo, replaceFrom, insertTo, insertFrom
    prefs_default =        [1.0] * 9
    prefs_insert_early =   [1.0, 1.0, 1.0, 1.5, 1.5, 1.2, 1.2, 1.3, 1.3]
    prefs_remove_late =    [1.0, 1.0, 1.0, 1.0, 1.5, 1.2, 1.2, 1.0, 1.0]
    prefs_insert_connected=[1.0, 1.0, 1.0, 1.0, 1.5, 1.2, 1.2, 1.5, 1.5]
    prefs_list = [prefs_default, prefs_insert_early, prefs_remove_late, prefs_insert_connected]
    
    with Pool(processes=amount) as pool:
        tasks = []                         
        for i in range(amount):
            tasks.append([start_solution_path, graph, i, nodes, prefs_list[i]])

        results = pool.map(sa_runner, tasks)
    return results

def test(solution:list, graph:list):
    """
    Test if the solution is a valid path
    """
    for i in range(len(solution) - 1):
        a, b = solution[i], solution[i + 1]
        wei = graph[a][b]
        if wei == -1: 
            print("invalid")
            return False
    return True

def dijkstra(graph: list[list[int]], origin: int, target: int) -> list[int]:
    """
    Using dijkstra to find the shortest path to test the quality of the path found by SA and FW"
    """
    n = len(graph)
    distances = [float('inf')] * n
    prev = [None] * n
    visited = [False] * n

    distances[origin] = 0
    heap = [(0, origin)]
    
    target_wei = -1

    while heap:
        cost, start = heapq.heappop(heap)

        if visited[start]:
            continue
        visited[start] = True

        for end, weight in enumerate(graph[start]):
            if weight<1: continue

            if distances[start] + weight < distances[end]:
                if end==target: target_wei = distances[start] + weight
                distances[end] = distances[start] + weight
                prev[end] = start
                heapq.heappush(heap, (distances[end], end))

    # Reconstruct the shortest path
    if distances[target] == float('inf'):
        return [], target_wei # No path

    path = []
    current = target
    while current is not None:
        path.append(current)
        current = prev[current]
    path.reverse()
    return path, target_wei

def convertToVisualizationFormat(path_list:list, id_dict:dict, sa_num:int, fw_path:list, optimal_path:list, sa_path:list)->None:
    visual_list = []
    for best in path_list:
        id_list = []
        for node in best:
            id_list.append(id_dict[node])
        visual_list.append(id_list)
        
    fw_id = []
    optimal_id = []
    sa_id = []
    
    for node in fw_path:
        fw_id.append(id_dict[node])
    for node in optimal_path:
        optimal_id.append(id_dict[node])
    for node in sa_path:
        sa_id.append(id_dict[node])
        
    joblib.dump((visual_list, fw_id, sa_id, optimal_id), f'saved_path_{sa_num}.joblib')

def dfs(graph: list[list[int]], origin: int, target: int) -> list:
    stack = [(origin, [origin])]
    visited = set()

    while stack:
        node, path = stack.pop()

        if node == target:
            return path

        if node in visited:
            continue
        visited.add(node)

        for neighbor, weight in enumerate(graph[node]):
            if weight > 0 and neighbor not in path:
                stack.append((neighbor, path + [neighbor]))

    return []  # No path found

def convertToOSMID(path:list, id_dict:list):
    id_path = []
    for node in path:
        id_path.append(id_dict[node])
    return id_path

if __name__ == "__main__":   
    
    radius = 300
    vn = 10.767131408508137, 106.67267902509788
    
    try: graph, original_id_dict = joblib.load('saved_graph.joblib')
    except FileNotFoundError: 
        # change this variable to change the graph size       
        print("Generating adjacency graph....\n")
        generateGraph(radius, vn)
        graph, original_id_dict = joblib.load('saved_graph.joblib') 
    
    
    nodes = len(graph)
    print(f"Number of nodes: {nodes}\n")
    maxWei = 2000 # remember to updates getCost when change max weight    
    sa_solvers = 4
    fw_wei = maxWei//5

    origin, target = sample(range(0, len(graph)), 2)
    print("origin:", origin, "target:", target)
    print()
    
    if nodes<101:
        tracemalloc.start()
        start = time()
        fw_graph, next_node, fw_changed_graph = floydWarshall(deepcopy(graph), get_max_iterations(nodes))

        fw_path, fw_wei = reconstruct_path(origin, target, next_node, fw_changed_graph)
        if fw_wei!=-1:
            print("FW time:", round(time()-start, 2), "s")
            fw_current, fw_peak = tracemalloc.get_traced_memory()
            print(f"Peak memory usage: {fw_peak / 1024:.2f} KB")
            print()
            
            print("FW path:",fw_path)
            print("FW weight:", fw_wei)
            print()
            
            fw_optimal_path, optimal_wei = dijkstra(fw_changed_graph, origin, target)
            quality = round((abs(fw_wei - optimal_wei) / optimal_wei)*100, 2) # how close to the optimal
            print()
            print("FW error:", quality, "%")
            print("\n----------------------------------\n")

            
        tracemalloc.stop()
        tracemalloc.clear_traces()

    tracemalloc.start()
    start = time()
    others = set()
    start_solution_path= dfs(graph, origin, target)
    if len(start_solution_path)==0:
        print(f"cannot find connection between {origin} and {target}")
    else:
        sa_result = run_parallel_sa(graph, nodes, start_solution_path, sa_solvers)
        best_sa_wei = float('inf')
        for id, res in enumerate(sa_result):
            if test(res[0], graph):
                if res[3]<best_sa_wei:
                    best_sa_result = (res[0], res[1], res[3], res[4], res[6])
                    best_sa_wei = res[3]
                    
        print("SA time:", round(time()-start, 2), "s")
        sa_current, sa_peak = tracemalloc.get_traced_memory()
        print(f"Peak memory usage: {sa_peak / 1024:.2f} KB")
        print()
        
        print("SA path:", best_sa_result[0])
        print("SA weight:", best_sa_result[1])
        print()
        
        sa_optimal_path, optimal_wei = dijkstra(best_sa_result[3], origin, target)
        quality = round((abs(best_sa_result[1] - optimal_wei) / optimal_wei)*100, 2) # relative error
        improvement = round(((best_sa_result[2] - best_sa_result[1]) / best_sa_result[2])* 100, 2)
        print("SA error:", quality, "%")
        print("SA improvement:", improvement, "%")
        print("SA Fast Convergence Score:", round(best_sa_result[4], 2), "%")
        
        print("\nPlotting (may take for minutes)....")
        plot_paths(
                    convertToOSMID(best_sa_result[0], original_id_dict),
                    convertToOSMID(sa_optimal_path, original_id_dict),
                    convertToOSMID(fw_path, original_id_dict),
                    convertToOSMID(fw_optimal_path, original_id_dict),
                    vn,
                    radius
                   )
        
    tracemalloc.stop()
    tracemalloc.clear_traces()
    
