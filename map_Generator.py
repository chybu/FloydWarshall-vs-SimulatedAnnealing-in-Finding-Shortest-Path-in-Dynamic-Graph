import osmnx as ox
import matplotlib.pyplot as plt
import joblib

def plot_paths(sim_anneal_path, sa_djk_path, floyd_warshall_path, fw_djk_path, location, radius):
    settings = {
        'start': {
            'color': 'green',
            'size': 50,  
            'marker': 'o',
            'label': 'Start',
            
            'linewidths': 1.5
        },
        'end': {
            'color': 'blue',
            'size': 50,
            'marker': 'X',
            'label': 'End',
            
            'linewidths': 1.5
        },
        'sim_anneal': {
            'color': 'red',
            'linewidth': 2, 
            'label': 'Simulated Annealing',
            'linestyle': '-',
            'alpha': 0.9
        },
        'sa_dijkstra': {
            'color': 'orange',
            'linewidth': 3,  # Thicker line
            'label': "Dijkstra of Simulated Annealing's",
            'linestyle': ":",
            'alpha': 0.9
        },
        'fw_dijkstra': {
            'color': 'orange',
            'linewidth': 3,  # Thicker line
            'label': "Dijkstra of Floyd-Warshall's",
            'linestyle': ":",
            'alpha': 0.9
        },
        'floyd_warshall': {
            'color': 'purple',
            'linewidth': 3,  # Thicker line
            'label': 'Floyd-Warshall',
            'linestyle': '--',
            'alpha': 0.9
        },
        'graph': {
            'node_size': 10,  # Make nodes visibl
            'edge_linewidth': 1  # Thicker streets
        }
    }
    
    start_node, end_node = sim_anneal_path[0], sim_anneal_path[-1]
    
    # Get Huntington graph
    G = ox.graph_from_point(location, dist=radius, dist_type="bbox", network_type="drive") 
    
    # Get coordinates for all paths
    def get_coords(path):
        return [(G.nodes[node]['x'], G.nodes[node]['y']) for node in path]
    
    sa_coords = get_coords(sim_anneal_path)
    sa_dj_coords = get_coords(sa_djk_path)
    fw_coords = get_coords(floyd_warshall_path)
    fw_dj_coords = get_coords(fw_djk_path)
    
    # Calculate the bounds to zoom in with dynamic padding
    all_x = [x for x, y in sa_coords + sa_dj_coords + fw_coords + fw_dj_coords]
    all_y = [y for x, y in sa_coords + sa_dj_coords + fw_coords + fw_dj_coords]
    
    # Calculate automatic padding based on the spread of points
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    padding = max(x_range, y_range) * 0.15  # 15% padding
    
    x_min, x_max = min(all_x) - padding, max(all_x) + padding
    y_min, y_max = min(all_y) - padding, max(all_y) + padding
    
    # Create figure with four subplots in 2x2 grid
    fig, axs = plt.subplots(2, 3, figsize=(16, 16))
    fig.suptitle('Path Finding Algorithm Comparison', 
                fontsize=13, y=0.98, weight='bold')
    # Get street names for labeling
    street_names = {}
    for u, v, key, data in G.edges(keys=True, data=True):
        if 'name' in data:
            # Get midpoint of the edge for label placement
            x = (G.nodes[u]['x'] + G.nodes[v]['x']) / 2
            y = (G.nodes[u]['y'] + G.nodes[v]['y']) / 2
            street_names[(x, y)] = data['name']
            
    # Common plot settings for all subplots
    for ax in axs.flat:
        used_street_names = set()
        ox.plot_graph(G, ax=ax, show=False, close=False,
                     node_size=settings['graph']['node_size'],
                     edge_linewidth=settings['graph']['edge_linewidth'])
        for (x, y), name in street_names.items():
            # Only plot if within visible bounds
            if isinstance(name, list): name = name[0]
            if x_min <= x <= x_max and y_min <= y <= y_max and name not in used_street_names:
                used_street_names.add(name)
                ax.text(x, y, name, fontsize='small', color='darkslategray',
                           alpha=0.7, ha='center', va='center', )
                   
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, linestyle=':', alpha=0.7)
        ax.set_facecolor('#f5f5f5')  # Light gray background
    
    # Plot 1.1: All paths overlaid
    ax1 = axs[0, 0]
    ax1.set_title("Floyd-Warshall and Dijkstra", pad=12, weight='bold')
    
    # Plot order matters for visibility - first plotted is on bottom
    # Floyd-Warshall (bottom layer)
    ax1.plot([x for x, y in fw_coords], [y for x, y in fw_coords], 
             color=settings['floyd_warshall']['color'], 
             linewidth=settings['floyd_warshall']['linewidth'],
             linestyle=settings['floyd_warshall']['linestyle'],
             alpha=settings['floyd_warshall']['alpha'],
             label=settings['floyd_warshall']['label'])
    
    # Dijkstra's (top layer)
    ax1.plot([x for x, y in fw_dj_coords], [y for x, y in fw_dj_coords], 
             color=settings['fw_dijkstra']['color'], 
             linewidth=settings['fw_dijkstra']['linewidth'],
             linestyle=settings['fw_dijkstra']['linestyle'],
             alpha=settings['fw_dijkstra']['alpha'],
             label=settings['fw_dijkstra']['label'])
    
    # Plot start and end points
    ax1.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
                c=settings['start']['color'],
                s=settings['start']['size'],
                marker=settings['start']['marker'],
                
                linewidths=settings['start']['linewidths'],
                zorder=5,  # Highest z-order
                label=settings['start']['label'])
    
    ax1.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'],
                c=settings['end']['color'],
                s=settings['end']['size'],
                marker=settings['end']['marker'],
            
                linewidths=settings['end']['linewidths'],
                zorder=5,
                label=settings['end']['label'])
    
    ax1.set_xlabel("Longitude", labelpad=10)
    ax1.set_ylabel("Latitude", labelpad=10)
    ax1.legend(loc='upper right', bbox_to_anchor=(-0.05, 1), borderaxespad=0., framealpha=1, edgecolor='black', fontsize=9)   
    
    # Plot 1.2: Just Floyd-Warshall
    ax3 = axs[0, 1]
    ax3.set_title("Floyd-Warshall Path", pad=12, weight='bold')
    ax3.plot([x for x, y in fw_coords], [y for x, y in fw_coords], 
             color=settings['floyd_warshall']['color'], 
             linewidth=settings['floyd_warshall']['linewidth']+1,
             linestyle=settings['floyd_warshall']['linestyle'],
             alpha=0.95)
    
    ax3.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
                c=settings['start']['color'],
                s=settings['start']['size']*1.2,
                marker=settings['start']['marker'],
                
                linewidths=settings['start']['linewidths'],
                zorder=5)
    
    ax3.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'],
                c=settings['end']['color'],
                s=settings['end']['size']*1.2,
                marker=settings['end']['marker'],
                
                linewidths=settings['end']['linewidths'],
                zorder=5)
    
    ax3.set_xlabel("Longitude", labelpad=10)
    ax3.set_ylabel("Latitude", labelpad=10)
    
    # Plot 1.3: Just Dijkstra's
    ax4 = axs[0, 2]
    ax4.set_title("Dijkstra of Floyd-Warshall's Path", pad=12, weight='bold')
    ax4.plot([x for x, y in fw_dj_coords], [y for x, y in fw_dj_coords], 
             color=settings['fw_dijkstra']['color'], 
             linewidth=settings['fw_dijkstra']['linewidth']+1,
             linestyle=settings['fw_dijkstra']['linestyle'],
             alpha=0.95)
    
    ax4.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
                c=settings['start']['color'],
                s=settings['start']['size']*1.2,
                marker=settings['start']['marker'],
                
                linewidths=settings['start']['linewidths'],
                zorder=5)
    
    ax4.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'],
                c=settings['end']['color'],
                s=settings['end']['size']*1.2,
                marker=settings['end']['marker'],
                
                linewidths=settings['end']['linewidths'],
                zorder=5)
    
    ax4.set_xlabel("Longitude", labelpad=10)
    ax4.set_ylabel("Latitude", labelpad=10)
    
    
    # Plot 2.1: All paths overlaid
    ax1 = axs[1, 0]
    ax1.set_title("Simulated Annealing and Dijkstra", pad=12, weight='bold')
    
    # Plot order matters for visibility - first plotted is on bottom
    # Simmulated Annealing (bottom layer)
    ax1.plot([x for x, y in sa_coords], [y for x, y in sa_coords], 
             color=settings['sim_anneal']['color'], 
             linewidth=settings['sim_anneal']['linewidth'],
             linestyle=settings['sim_anneal']['linestyle'],
             alpha=settings['sim_anneal']['alpha'],
             label=settings['sim_anneal']['label'])
    
    # Dijkstra's (top layer)
    ax1.plot([x for x, y in sa_dj_coords], [y for x, y in sa_dj_coords], 
             color=settings['fw_dijkstra']['color'], 
             linewidth=settings['fw_dijkstra']['linewidth'],
             linestyle=settings['fw_dijkstra']['linestyle'],
             alpha=settings['fw_dijkstra']['alpha'],
             label=settings['sa_dijkstra']['label'])
    
    # Plot start and end points
    ax1.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
                c=settings['start']['color'],
                s=settings['start']['size'],
                marker=settings['start']['marker'],
                
                linewidths=settings['start']['linewidths'],
                zorder=5,  # Highest z-order
                label=settings['start']['label'])
    
    ax1.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'],
                c=settings['end']['color'],
                s=settings['end']['size'],
                marker=settings['end']['marker'],
            
                linewidths=settings['end']['linewidths'],
                zorder=5,
                label=settings['end']['label'])
    
    ax1.set_xlabel("Longitude", labelpad=10)
    ax1.set_ylabel("Latitude", labelpad=10)
    
    # Plot 2.2: Just Simulated Annealing
    ax2 = axs[1, 1]
    ax2.set_title("Simulated Annealing Path", pad=12, weight='bold')
    ax2.plot([x for x, y in sa_coords], [y for x, y in sa_coords], 
             color=settings['sim_anneal']['color'], 
             linewidth=settings['sim_anneal']['linewidth']+1,  # Even thicker for single path
             linestyle=settings['sim_anneal']['linestyle'],
             alpha=0.95)
    
    ax2.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
                c=settings['start']['color'],
                s=settings['start']['size']*1.2,  # Larger markers
                marker=settings['start']['marker'],
        
                linewidths=settings['start']['linewidths'],
                zorder=5)
    
    ax2.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'],
                c=settings['end']['color'],
                s=settings['end']['size']*1.2,
                marker=settings['end']['marker'],
                
                linewidths=settings['end']['linewidths'],
                zorder=5)
    
    ax2.set_xlabel("Longitude", labelpad=10)
    ax2.set_ylabel("Latitude", labelpad=10)
    
    # Plot 2.3: Just Dijkstra's
    ax4 = axs[1, 2]
    ax4.set_title("Dijkstra of Simulated Annealing's Path", pad=12, weight='bold')
    ax4.plot([x for x, y in sa_dj_coords], [y for x, y in sa_dj_coords], 
             color=settings['fw_dijkstra']['color'], 
             linewidth=settings['fw_dijkstra']['linewidth']+1,
             linestyle=settings['fw_dijkstra']['linestyle'],
             alpha=0.95)
    
    ax4.scatter(G.nodes[start_node]['x'], G.nodes[start_node]['y'],
                c=settings['start']['color'],
                s=settings['start']['size']*1.2,
                marker=settings['start']['marker'],
                
                linewidths=settings['start']['linewidths'],
                zorder=5)
    
    ax4.scatter(G.nodes[end_node]['x'], G.nodes[end_node]['y'],
                c=settings['end']['color'],
                s=settings['end']['size']*1.2,
                marker=settings['end']['marker'],
                
                linewidths=settings['end']['linewidths'],
                zorder=5)
    
    ax4.set_xlabel("Longitude", labelpad=10)
    ax4.set_ylabel("Latitude", labelpad=10)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    # plt.tight_layout()
    plt.subplots_adjust(hspace=0.25, wspace=0.15, top=0.85)
    plt.show()
    
    
