from cvxpy import *
import numpy as np
import sys
from collections import defaultdict
import networkx as nx

def populate_arcs(arcs, inflow_idxs, outflow_idxs, num_nodes):
  arc_id = 0
  for i in xrange(num_nodes):
    for j in xrange(num_nodes):
      if i != j:
        # From i->j
        arcs[(i, j)] = arc_id
        inflow_idxs[j].append(arc_id)
        outflow_idxs[i].append(arc_id)
        arc_id = arc_id + 1

def read_arcs(arcs, inflow_idxs, outflow_idxs, file_name, num_nodes):
  arc_id = 0
  with open(file_name) as f:
    for line in f:
      arc = line.split(" ")
      src = int(arc[0])
      dst = int(arc[1])
      assert src < num_nodes and dst < num_nodes
      arcs[(src, dst)] = arc_id
      inflow_idxs[dst].append(arc_id)
      outflow_idxs[src].append(arc_id)
      arc_id = arc_id + 1

def compute_cycles(arcs, cycles, num_nodes):
    G = nx.DiGraph()
    G.add_nodes_from(range(num_nodes))
    G.add_edges_from(arcs.keys())
    nx_cycs = nx.simple_cycles(G)
    for cyc in nx_cycs:
        cyc_edges = []
        # cyc if of format node1, node2 ..
        first_node = cyc[0]
        for i in xrange(len(cyc)):
            if i == len(cyc) - 1:
                cyc_edges.append(arcs[(cyc[i], first_node)])
            else:
                cyc_edges.append(arcs[(cyc[i], cyc[i + 1])])

        # Add cyc_edges to cycles
        cycles.append(cyc_edges)

def populate_rev_arc_ids(arcs, rev_arcs):
  for a in arcs.keys():
    rev_arc_id = arcs[(a[1], a[0])]
    rev_arcs[arcs[a]] = rev_arc_id

def populate_supply(supply, src_node):
  supply[:] = 0.0
  supply[src_node, :] = 1.0

# Nodes can only forward elements they have received or have supply of
def forwarding_rule(flow, supply, inflow_idxs, outflow_idxs):
  # inflow_mat and outflow_mat start as single rows
  # They will become matrices of size num_nodes x num_partitions
  inflow_mat =  sum_entries(flow[inflow_idxs[0], :], axis=0)

  # Note: since we can send out the same data on many links
  # we take max_entries here. i.e. if we send it out on two link
  # we should get a value of 1 and not 2
  outflow_mat = max_entries(flow[outflow_idxs[0], :], axis=0)
  for a in xrange(1, supply.shape[0]):
    # For each incoming arc to 'a', we extract its partitions
    # Then we sum up these rows to get all incoming partitions to 'a'
    inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
    inflow_mat = vstack(inflow_mat, inparts_node)

    outparts_node = max_entries(flow[outflow_idxs[a], :], axis=0)
    outflow_mat = vstack(outflow_mat, outparts_node)

  # Constraint is that element-wise
  # supply + out_flow - in_flow >= 0
  return ((supply + inflow_mat - outflow_mat) >= 0)

# All nodes should have all partitions eventually
# We enforce this by constructing matrices of all partitions that are
# supplied and all partitions that are input to a node
#
# The sum of these two matrices should have 1 in all elements
def broadcast_rule(flow, supply, inflow_idxs):
  inflow_mat =  sum_entries(flow[inflow_idxs[0], :], axis=0)
  for a in xrange(1, supply.shape[0]):
    # For each incoming arc to 'a', we extract its partitions
    # Then we sum up these rows to get all incoming partitions to 'a'
    inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
    inflow_mat = vstack(inflow_mat, inparts_node)

  return ((supply + inflow_mat) == 1)


def nocycles(flow, cycles):
    constraints = []
    for cyc in cycles:
        # This is a num_edges_in_cycle x num_partitions
        cyc_flow = flow[cyc, :]
        # Compute column sums and constraint the sum to be less than
        # number of edges
        constraints.append(sum_entries(cyc_flow, axis=0) <= (len(cyc) - 1))

    return constraints

# We shouldn't send and recv same partition on same link
# So for each arc find the reverse arc and make sure their
# sum is <= 1 for all partitions
def send_recv_same_link(flow, arcs, rev_arcs):
  constraints = []
  for a in arcs:
    # Get partitions on this arc and on this reverse arc
    arc_id = arcs[a]
    rev_arc_id = rev_arcs[arc_id]
    constraint = ((flow[arc_id, :] + flow[rev_arc_id, :]) <= 1)
    constraints.append(constraint)
  return constraints

def source_sends_all_parts(flow, source_node, outflow_idxs):
    # Extract all outgoing edges for source
    # This is a matrix of size num_out_edges x num_partitions
    outflow_mat = flow[outflow_idxs[source_node], :]
    # This is 1 x num_partitions
    outflow_entries = sum_entries(outflow_mat, axis=0)
    # Every partition needs to be sent. So constraint outflow_entries >= 1
    return outflow_entries >= 1

# def objective_fn(flow, inflow_idxs, supply):
#   inflow_mat =  sum_entries(flow[inflow_idxs[0], :], axis=0)
#   for a in xrange(1, supply.shape[0]):
#     # For each incoming arc to 'a', we extract its partitions
#     # Then we sum up these rows to get all incoming partitions to 'a'
#     inparts_node = sum_entries(flow[inflow_idxs[a], :], axis=0)
#     inflow_mat = vstack(inflow_mat, inparts_node)
#
#   return max_entries(sum_entries(inflow_mat, axis=1))

def node_load_objective_fn(flow, inflow_idxs):
  # Our objective is to compute the slowest node

  # To do this we find out the load on each link by summing entries on it
  # This is links x 1 matrix
  link_loads = sum_entries(flow, axis=1)

  # Now we build a nodes x 1 matrix 
  node_loads = max_entries(link_loads[inflow_idxs[0], :], axis=0)
  for a in xrange(1, len(inflow_idxs)):
    node_load = max_entries(link_loads[inflow_idxs[a], :], axis=0)
    node_loads = vstack(node_loads, node_load)

  # We then get the max value from the node_loads to get slowest node.
  return max_entries(node_loads)

def link_load_objective_fn(flow, inflow_idxs):
  return max_entries(sum_entries(flow, axis=1))

def main():

  if len(sys.argv) != 4:
    print "Usage cvxpy_test.py <num_nodes> <num_partitions> <arc file>"
    sys.exit(0)

  num_nodes = int(sys.argv[1])
  num_partitions = int(sys.argv[2])
  print "num_nodes ", num_nodes, " num_partitions ", num_partitions

  arc_filename = sys.argv[3]
  print "read topology file ", arc_filename
  arcs = {} # Map from (src, dest) -> arc_id
  rev_arcs = {} # Map from arc_id -> rev_arc_id
  cycles = [] # List of cycles. Each cycle is of the form [arc_id1, arc_id2, ...]

  inflow_idxs = defaultdict(list)
  outflow_idxs = defaultdict(list)
  supply = np.zeros((num_nodes, num_partitions))

  #populate_arcs(arcs, inflow_idxs, outflow_idxs, num_nodes)
  read_arcs(arcs, inflow_idxs, outflow_idxs, arc_filename, num_nodes)
  compute_cycles(arcs, cycles, num_nodes)

  populate_supply(supply, 0)

  flow = Bool(len(arcs), num_partitions)

  objective = Minimize(link_load_objective_fn(flow, inflow_idxs))
  constraints = [
      forwarding_rule(flow, supply, inflow_idxs, outflow_idxs),
      broadcast_rule(flow, supply, inflow_idxs)
  ]

  #constraints.append(source_sends_all_parts(flow, 0, outflow_idxs))
  constraints.extend(nocycles(flow, cycles))

  problem = Problem(objective, constraints)
  problem.solve(verbose=False)
  print "status:", problem.status
  print "optimal value", problem.value

  print "total time ", problem.value / num_partitions
  final_flow = flow.value

  print "Source -> Dest : Weights"
  for arc in sorted(arcs.keys()):
    arc_id = arcs[arc]
    print arc[0], "->", arc[1], ":", (np.round(final_flow[arc_id, :]))

# added by Guanhua, print out to .txt file
  print "writing generated scheme to dir.csv"
  with open("dir.csv",'ab') as text_file:
    for arc in sorted(arcs.keys()):
      arc_id = arcs[arc]
      text_file.write("%s,%s," %(arc[0],arc[1]))
      np.savetxt(text_file,np.absolute(np.round(final_flow[arc_id, :])),delimiter=",",newline='\n')
  

if __name__ == "__main__":
  main()
