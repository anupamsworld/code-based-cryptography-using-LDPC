import networkx as nx
import matplotlib.pyplot as plt
'''
# Create a graph
G = nx.Graph()

# Add nodes and edges
G.add_nodes_from(['A', 'B', 'C', 'D', 'E'])
G.add_edges_from([('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'), ('C', 'E')])

# Draw the graph
nx.draw(G, with_labels=True, node_color='lightblue', node_size=500, edge_color='gray', font_size=10)

# Display the plot
plt.show()
'''


# Create a bipartite graph
B = nx.Graph()

# Add nodes to the graph, specifying their bipartite membership
B.add_nodes_from(['A', 'B', 'C'], bipartite=0)  # Set A
B.add_nodes_from(['X', 'Y', 'Z'], bipartite=1)  # Set B

# Add edges between nodes from different sets
B.add_edges_from([('A', 'X'), ('A', 'Y'), ('B', 'Y'), ('B', 'Z'), ('C', 'X'), ('C', 'Z')])

# Create a layout for the bipartite graph
pos = nx.bipartite_layout(B, ['A', 'B', 'C'])

# Draw the graph
nx.draw(B, pos=pos, with_labels=True, node_color=['red', 'green'], node_size=1000, font_size=12)

# Show the plot
plt.show()