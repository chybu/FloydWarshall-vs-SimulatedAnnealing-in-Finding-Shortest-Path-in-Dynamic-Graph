import osmnx as ox
import pandas as pd
import joblib

def generateGraph(radius:int, location):
    """
    Generate adjacenct matrix
    """


    G = ox.graph_from_point(location, dist=radius, dist_type="bbox", network_type="drive") 

    # get a GeoSeries of consolidated intersections
    G_proj = ox.project_graph(G)
    # this reconnects edge geometries to the new consolidated nodes
    cleaned = ox.consolidate_intersections(G_proj, rebuild_graph=True, tolerance=15, dead_ends=False)

    # Convert to GeoDataFrames
    nodes, edges = ox.graph_to_gdfs(cleaned)

    edges = edges.reset_index()
    nodes = nodes.reset_index()
    edges['maxspeed_num'] = (
        edges['maxspeed']
        .astype(str)                      # Ensure everything is a string
        .str.extract(r'(\d+\.?\d*)')      # Regex: extract digits with optional decimal
        .astype(float)                    # Convert result to float
    )
    edges['lanes'] = pd.to_numeric(edges['lanes'], errors='coerce')
    edges['length'] = pd.to_numeric(edges['length'], errors='coerce')

    # Fill NaNs so comparisons don't break (set to 0 if unknown)
    edges['maxspeed_num'] = edges['maxspeed_num'].fillna(-1)
    edges['lanes'] = edges['lanes'].fillna(-1)
    edges['length'] = edges['length'].fillna(-1)

    # Sort by your criteria (descending: higher is better)
    edges = edges.sort_values(
        by=['u', 'v', 'length', 'maxspeed_num', 'lanes'],
        ascending=[True, True, False, False, False]
    )
    # Keep only the best edge for each u-v pair
    edges = edges.drop_duplicates(subset=['u', 'v'], keep='first')

    edges = edges[
        ~((edges['maxspeed_num'] == -1) & 
        (edges['length'] == -1) & 
        (edges['lanes'] == -1))
    ]

    edges["weight"] = (1/edges['lanes'] * (3 * edges['length']) * 1/edges['maxspeed_num']).round().abs().astype(int)
    max_weight = edges['weight'].max()

    number_of_nodes = nodes.shape[0]

    graph = []
    for i in range(number_of_nodes):
        graph.append([-1]*number_of_nodes)

    for i in range(number_of_nodes):
        graph[i][i] = 0
            
    for row in edges.itertuples(index=False):
        origin = row.u
        destination = row.v
        # length = row.length
        # maxspeed = row.maxspeed_num
        # lanes = row.lanes
        normalized_weight = int(round(1000*(row.weight/max_weight)))
        # print(normalized_weight, length, maxspeed, lanes)
        graph[origin][destination] = normalized_weight
        
    original_id_dict = dict()
    for row in nodes.itertuples(index=False):
        id = row.osmid
        original_id = row.osmid_original
        if type(original_id)!=int: original_id = original_id[0]
        original_id_dict[id] = original_id

    ed = edges.shape[0]
    nod = nodes.shape[0]
    dens = ed/(nod*(nod-1))


    print("Density:", round(dens,2))

    joblib.dump((graph, original_id_dict), 'saved_graph.joblib')
    # ox.plot_graph(cleaned, node_color="r", figsize=(10,10))
    
if __name__=="__main__":
    vn = 10.767131408508137, 106.67267902509788
    generateGraph(300, vn)