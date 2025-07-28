import random
import math
import pygame
import sys # Import sys for clean exit
import pickle # Import the pickle module for serialization

# --- Global Innovation Tracker ---
# This dictionary will keep track of unique innovations (new connections or nodes)
# to ensure that identical structural changes across different genomes get the same
# "innovation number". This is crucial for crossover and genetic distance calculation.
# Format: { (from_node_id, to_node_id): innovation_number, ... } for connections
#         { (old_connection_innovation_id, new_node_id): innovation_number, ... } for nodes
_innovation_tracker = {
    'connection': {},
    'node': {},
    'next_innovation_number': 0
}

def get_next_innovation_number():
    """Returns the next available innovation number and increments the counter."""
    num = _innovation_tracker['next_innovation_number']
    _innovation_tracker['next_innovation_number'] += 1
    return num

def get_connection_innovation(from_node_id, to_node_id):
    """
    Gets or creates an innovation number for a new connection.
    Ensures that the same connection (from->to) always gets the same innovation number.
    """
    key = (from_node_id, to_node_id)
    if key not in _innovation_tracker['connection']:
        _innovation_tracker['connection'][key] = get_next_innovation_number()
    return _innovation_tracker['connection'][key]

def get_node_innovation(old_connection_innovation_id):
    """
    Gets or creates an innovation number for a new node created by splitting a connection.
    Ensures that splitting the same connection always gets the same innovation number for the new node.
    """
    key = old_connection_innovation_id
    if key not in _innovation_tracker['node']:
        _innovation_tracker['node'][key] = get_next_innovation_number()
    return _innovation_tracker['node'][key]

# --- Node and Connection Classes (Internal to NeuralNetwork) ---

class NodeGene:
    """Represents a single neuron (node) in the neural network."""
    def __init__(self, node_id, node_type, activation_function='sigmoid'):
        self.node_id = node_id
        self.node_type = node_type  # 'INPUT', 'HIDDEN', 'OUTPUT'
        self.activation_function = activation_function
        self.value = 0.0  # Current activation value
        self.input_sum = 0.0 # Sum of weighted inputs before activation

    def activate(self):
        """Applies the activation function to the input sum."""
        if self.node_type == 'INPUT':
            pass
        elif self.activation_function == 'sigmoid':
            self.value = self._sigmoid(self.input_sum)
        elif self.activation_function == 'relu':
            self.value = self._relu(self.input_sum)
        elif self.activation_function == 'tanh':
            self.value = self._tanh(self.input_sum)
        elif self.activation_function == 'leaky_relu': # New: Leaky ReLU activation
            self.value = self._leaky_relu(self.input_sum)
        else:
            self.value = self._sigmoid(self.input_sum)

    def _sigmoid(self, x):
        """Sigmoid activation function."""
        if x > 10: return 1.0
        if x < -10: return 0.0
        return 1.0 / (1.0 + self._exp(-x))

    def _relu(self, x):
        """ReLU activation function."""
        return max(0.0, x)

    def _tanh(self, x):
        """Tanh activation function."""
        if x > 10: return 1.0
        if x < -10: return -1.0
        return (self._exp(x) - self._exp(-x)) / (self._exp(x) + self._exp(-x))

    def _leaky_relu(self, x, alpha=0.01):
        """Leaky ReLU activation function."""
        return x if x > 0 else alpha * x

    def _exp(self, x):
        """
        Custom exponential function approximation.
        This is a very basic approximation and might not be precise for all values.
        For a real application, a more robust `math.exp` would be used.
        """
        res = 1.0
        term = 1.0
        for i in range(1, 15): # Iterate for a few terms
            term *= x / i
            res += term
            if abs(term) < 1e-9: # Stop if term is too small
                break
        return res

    def clone(self):
        """Creates a deep copy of the NodeGene."""
        return NodeGene(self.node_id, self.node_type, self.activation_function)

    def __repr__(self):
        return f"Node(ID:{self.node_id}, Type:{self.node_type}, Act:{self.activation_function}, Val:{self.value:.2f})"

class ConnectionGene:
    """Represents a connection (synapse) between two neurons."""
    def __init__(self, from_node_id, to_node_id, weight, enabled, innovation_number):
        self.from_node_id = from_node_id
        self.to_node_id = to_node_id
        self.weight = weight
        self.enabled = enabled
        self.innovation_number = innovation_number # Unique ID for this connection gene

    def clone(self):
        """Creates a deep copy of the ConnectionGene."""
        return ConnectionGene(self.from_node_id, self.to_node_id, self.weight, self.enabled, self.innovation_number)

    def __repr__(self):
        return (f"Conn(Inn:{self.innovation_number}, From:{self.from_node_id} "
                f"To:{self.to_node_id}, W:{self.weight:.2f}, Enabled:{self.enabled})")

# --- NeuralNetwork Class (Genome) ---

class NeuralNetwork:
    """
    Represents a single neural network genome in the NEAT algorithm.
    It manages its nodes, connections, and handles the forward pass.
    """
    def __init__(self, input_size=None, output_size=None, initialize_nodes=True):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}  # {node_id: NodeGene_object}
        self.connections = {} # {innovation_number: ConnectionGene_object}
        self.node_id_counter = 0 # To assign unique IDs to new nodes

        self.fitness = 0.0 # Placeholder for fitness value

        if initialize_nodes and input_size is not None and output_size is not None:
            self._initialize_network()

    def _initialize_network(self, initial_connection_prob=0.1):
        """
        Initializes the basic structure with input and output nodes,
        and optionally adds sparse connections between them.
        """
        input_node_ids = []
        hidden_node_ids = []
        output_node_ids = []

        # Create input nodes
        for _ in range(self.input_size):
            node_id = self._get_next_node_id()
            self.nodes[node_id] = NodeGene(node_id, 'INPUT')
            input_node_ids.append(node_id)

        # Create a random number of hidden nodes (between 0 and 5)
        num_hidden_to_add = random.randint(0, 5)
        for _ in range(num_hidden_to_add):
            node_id = self._get_next_node_id()
            self.nodes[node_id] = NodeGene(node_id, 'HIDDEN', activation_function='leaky_relu') # Set leaky_relu for hidden nodes
            hidden_node_ids.append(node_id)

        # Create output nodes
        for _ in range(self.output_size):
            node_id = self._get_next_node_id()
            self.nodes[node_id] = NodeGene(node_id, 'OUTPUT')
            output_node_ids.append(node_id)

        # Ensure each input neuron is connected to at least one hidden or output neuron
        all_non_input_node_ids = hidden_node_ids + output_node_ids
        if all_non_input_node_ids: # Only proceed if there are nodes to connect to
            for input_id in input_node_ids:
                # Try to connect to a random non-input node
                target_node_id = random.choice(all_non_input_node_ids)
                self.add_connection(input_id, target_node_id)

        # Add initial sparse connections between input and output/hidden nodes
        # This covers cases where the above loop might create only one connection per input
        # and also adds connections between hidden and output nodes.
        for from_node_id in input_node_ids + hidden_node_ids:
            for to_node_id in hidden_node_ids + output_node_ids:
                if from_node_id == to_node_id: # Skip self-loops
                    continue
                # Prevent input to input, output to input/hidden
                if self.nodes[from_node_id].node_type == 'INPUT' and self.nodes[to_node_id].node_type == 'INPUT':
                    continue
                if self.nodes[from_node_id].node_type == 'OUTPUT' and self.nodes[to_node_id].node_type in ['INPUT', 'HIDDEN']:
                    continue

                if random.random() < initial_connection_prob:
                    self.add_connection(from_node_id, to_node_id)

        # --- NEW LOGIC: Ensure all hidden neurons are connected to at least one output neuron ---
        if hidden_node_ids and output_node_ids: # Only if there are hidden and output nodes
            for hidden_id in hidden_node_ids:
                # Check if this hidden node has any outgoing connections to an output node
                has_output_connection = False
                for conn in self.connections.values():
                    if conn.from_node_id == hidden_id and self.nodes[conn.to_node_id].node_type == 'OUTPUT':
                        has_output_connection = True
                        break
                
                if not has_output_connection:
                    # If no outgoing connection to an output node, add one
                    target_output_id = random.choice(output_node_ids)
                    self.add_connection(hidden_id, target_output_id)
        # --- END NEW LOGIC ---

        # Apply 3 random mutations to introduce initial diversity
        for _ in range(3):
            self._mutate_initial() # Use a special initial mutation method if needed, or just _mutate

    def _mutate_initial(self):
        """
        A simplified mutation method for initial network setup,
        focuss_on adding connections and nodes.
        """
        # Prioritize adding connections first
        if random.random() < 0.7: # Higher chance to add connection
            all_node_ids = list(self.nodes.keys())
            if len(all_node_ids) >= 2:
                from_node_id = random.choice(all_node_ids)
                to_node_id = random.choice(all_node_ids)
                self.add_connection(from_node_id, to_node_id)
        else: # Then add node
            enabled_conns = [c for c in self.connections.values() if c.enabled]
            if enabled_conns:
                conn_to_split = random.choice(enabled_conns)
                self.add_node(conn_to_split.innovation_number)
        
        # Also apply some weight mutation
        if self.connections:
            conn = random.choice(list(self.connections.values()))
            conn.weight += random.gauss(0, 0.5)
            conn.weight = max(-5.0, min(5.0, conn.weight))


    def _get_next_node_id(self):
        """Returns a unique ID for a new node."""
        node_id = self.node_id_counter
        self.node_id_counter += 1
        return node_id

    def add_node(self, connection_innovation_id):
        """
        Adds a new node to the network by splitting an existing, enabled connection.
        The old connection is disabled, and two new connections are added:
        (from_node -> new_node) with weight 1.0
        (new_node -> to_node) with the old connection's weight.
        """
        if connection_innovation_id not in self.connections:
            return False

        old_conn = self.connections[connection_innovation_id]
        if not old_conn.enabled:
            return False

        # Disable the old connection
        old_conn.enabled = False

        # Create new node
        new_node_id = self._get_next_node_id()
        new_node_innovation = get_node_innovation(old_conn.innovation_number) # Get innovation for the node itself
        self.nodes[new_node_id] = NodeGene(new_node_id, 'HIDDEN', activation_function='leaky_relu') # Set leaky_relu for new hidden nodes
        
        # Create first new connection (from old_conn.from_node_id to new_node_id)
        conn1_innovation = get_connection_innovation(old_conn.from_node_id, new_node_id)
        self.connections[conn1_innovation] = ConnectionGene(
            old_conn.from_node_id, new_node_id, 1.0, True, conn1_innovation
        )

        # Create second new connection (from new_node_id to old_conn.to_node_id)
        conn2_innovation = get_connection_innovation(new_node_id, old_conn.to_node_id)
        self.connections[conn2_innovation] = ConnectionGene(
            new_node_id, old_conn.to_node_id, old_conn.weight, True, conn2_innovation
        )
        return True

    def add_connection(self, from_node_id, to_node_id, weight=None):
        """
        Adds a new connection between two existing nodes.
        Ensures the connection doesn't already exist and is not a self-loop.
        Also prevents connections from output to input/hidden, or input to input.
        """
        if from_node_id == to_node_id: # No self-loops initially
            return False
        if from_node_id not in self.nodes or to_node_id not in self.nodes:
            return False

        from_node_type = self.nodes[from_node_id].node_type
        to_node_type = self.nodes[to_node_id].node_type

        # Prevent invalid connections (e.g., output to input/hidden, input to input)
        if from_node_type == 'OUTPUT' and to_node_type in ['INPUT', 'HIDDEN']:
            return False
        if from_node_type == 'INPUT' and to_node_type == 'INPUT':
            return False

        # Check if connection already exists (enabled or disabled)
        for conn in self.connections.values():
            if conn.from_node_id == from_node_id and conn.to_node_id == to_node_id:
                if not conn.enabled: # If disabled, re-enable it instead of adding new
                    conn.enabled = True
                    return True
                return False # Already exists and enabled

        # Get innovation number for this potential new connection
        innovation_num = get_connection_innovation(from_node_id, to_node_id)

        # Assign a random weight if not provided
        if weight is None:
            weight = random.uniform(-1.0, 1.0)

        self.connections[innovation_num] = ConnectionGene(
            from_node_id, to_node_id, weight, True, innovation_num
        )
        return True

    def get_genes(self):
        """Returns the nodes and connections as lists, suitable for crossover."""
        return list(self.nodes.values()), list(self.connections.values())

    def set_genes(self, nodes, connections):
        """Sets the network's structure from provided node and connection genes."""
        self.nodes = {node.node_id: node for node in nodes}
        self.connections = {conn.innovation_number: conn for conn in connections}
        # Update node_id_counter to ensure new nodes get unique IDs
        if self.nodes:
            self.node_id_counter = max(node.node_id for node in self.nodes.values()) + 1
        else:
            self.node_id_counter = 0


    def _normalize_data(self, data, method='min_max', min_val=0, max_val=1):
        """
        Normalizes a list of numerical data using specified method.
        Args:
            data (list): List of numbers to normalize.
            method (str): 'min_max', 'z_score', or 'mean'.
            min_val (float): Target min for min_max.
            max_val (float): Target max for min_max.
        Returns:
            list: Normalized data.
        """
        if not data:
            return []

        if method == 'min_max':
            data_min = min(data)
            data_max = max(data)
            if data_max == data_min: # Avoid division by zero
                return [min_val + (max_val - min_val) / 2.0] * len(data) # Return middle of range
            return [min_val + (x - data_min) * (max_val - min_val) / (data_max - data_min) for x in data]
        elif method == 'z_score':
            data_mean = sum(data) / len(data)
            # Calculate standard deviation
            variance = sum([(x - data_mean)**2 for x in data]) / len(data)
            std_dev = variance**0.5
            if std_dev == 0: # Avoid division by zero
                return [0.0] * len(data)
            return [(x - data_mean) / std_dev for x in data]
        elif method == 'mean':
            data_mean = sum(data) / len(data)
            if data_mean == 0: # Avoid division by zero
                return [0.0] * len(data)
            return [x / data_mean for x in data]
        else:
            # No normalization, return original data
            return list(data)

    def forward(self, inputs, iterations=5, normalization_method='min_max'):
        """
        Performs a forward pass through the neural network.
        Handles recursive connections by iterating activation propagation.
        Args:
            inputs (list): List of input values.
            iterations (int): Number of iterations for recurrent networks to stabilize.
            normalization_method (str): Method to normalize input data ('min_max', 'z_score', 'mean', 'none').
        Returns:
            list: Output values from the network.
        """
        if len(inputs) != self.input_size:
            raise ValueError(f"Input size mismatch. Expected {self.input_size}, got {len(inputs)}")

        # Reset all node values and input sums
        for node in self.nodes.values():
            node.value = 0.0
            node.input_sum = 0.0

        # Set input node values
        input_nodes = [node for node in self.nodes.values() if node.node_type == 'INPUT']
        normalized_inputs = self._normalize_data(inputs, normalization_method)
        for i, node in enumerate(input_nodes):
            node.value = normalized_inputs[i]

        # Iteratively propagate activations for recurrent connections
        for _ in range(iterations):
            # Store previous values for recurrent connections
            prev_node_values = {node_id: node.value for node_id, node in self.nodes.items()}

            # Reset input sums for non-input nodes
            for node in self.nodes.values():
                if node.node_type != 'INPUT':
                    node.input_sum = 0.0

            # Calculate input sums for all nodes based on current values
            for conn in self.connections.values():
                if conn.enabled:
                    from_node = self.nodes.get(conn.from_node_id)
                    to_node = self.nodes.get(conn.to_node_id)
                    if from_node and to_node:
                        # Use previous value for the source node to avoid immediate feedback loops
                        # within the same iteration step.
                        to_node.input_sum += prev_node_values.get(from_node.node_id, 0.0) * conn.weight

            # Activate all non-input nodes
            for node in self.nodes.values():
                if node.node_type != 'INPUT':
                    node.activate()

        # Collect output node values
        output_nodes = sorted([node for node in self.nodes.values() if node.node_type == 'OUTPUT'], key=lambda n: n.node_id)
        return [node.value for node in output_nodes]

    def clone(self):
        """Creates a deep copy of the neural network."""
        # Create an empty NN, then copy over the specific nodes/connections
        new_nn = NeuralNetwork(self.input_size, self.output_size, initialize_nodes=False)
        new_nn.nodes = {node_id: node.clone() # Use clone method for NodeGene
                        for node_id, node in self.nodes.items()}
        new_nn.connections = {conn_id: conn.clone() # Use clone method for ConnectionGene
                              for conn_id, conn in self.connections.items()}
        new_nn.node_id_counter = self.node_id_counter
        new_nn.fitness = self.fitness
        return new_nn

    def __repr__(self):
        return (f"NN(Nodes:{len(self.nodes)}, Conns:{len(self.connections)}, "
                f"Enabled:{sum(1 for c in self.connections.values() if c.enabled)}, Fitness:{self.fitness:.2f})")

class Species:
    """Represents a single species in the NEAT algorithm."""
    def __init__(self, representative_genome):
        self.representative = representative_genome.clone() # The genome that defines this species
        self.members = [representative_genome] # List of genomes belonging to this species
        self.best_fitness_in_history = representative_genome.fitness # Best fitness ever achieved by a member of this species
        self.generations_stagnant = 0 # Number of generations without improvement in best_fitness_in_history
        self.age = 0 # How many generations this species has existed

    def add_member(self, genome):
        """Adds a genome to this species."""
        self.members.append(genome)

    def update_stagnation(self):
        """Updates the stagnation counter and best fitness for the species."""
        self.age += 1
        if not self.members: # If species is empty, it's stagnant
            self.generations_stagnant += 1
            return

        # Find the best fitness among current members
        current_best = max(g.fitness for g in self.members)
        if current_best > self.best_fitness_in_history:
            self.best_fitness_in_history = current_best
            self.generations_stagnant = 0 # Reset stagnation if improvement
        else:
            self.generations_stagnant += 1 # Increment if no improvement

    def reset_members(self):
        """Clears the members list for the next generation's speciation."""
        self.members = []

    def __len__(self):
        """Returns the number of members in the species."""
        return len(self.members)

    def __repr__(self):
        return (f"Species(RepID:{self.representative.node_id_counter}, Members:{len(self.members)}, "
                f"BestFit:{self.best_fitness_in_history:.2f}, Stagnant:{self.generations_stagnant}, Age:{self.age})")


# --- GeneticAlgorithm Class ---

class GeneticAlgorithm:
    """
    Implements the NEAT genetic algorithm to evolve neural networks.
    Manages populations, speciation, reproduction, and mutation.
    """
    def __init__(self,
                 num_populations=None, # New: total number of distinct populations
                 agents_per_population=None, # New: size of each sub-population
                 input_size=None,
                 output_size=None,
                 config=None,
                 initialize_populations=True): # New argument
        self.num_populations = num_populations
        self.agents_per_population = agents_per_population
        self.input_size = input_size
        self.output_size = output_size

        # Default configuration parameters
        self.config = {
            'c1': 1.0,  # Coefficient for excess genes
            'c2': 1.0,  # Coefficient for disjoint genes
            'c3': 0.4,  # Coefficient for weight differences
            'compatibility_threshold': 3.0, # Threshold for speciation
            'add_node_prob': 0.05, # Increased from 0.03
            'add_connection_prob': 0.08, # Increased from 0.05
            'weight_mutate_prob': 0.8,
            'weight_replace_prob': 0.1, # Probability to replace weight instead of perturbing
            'enable_disable_prob': 0.01, # Probability to toggle connection enabled status
            'inter_species_breeding_prob': 0.001, # Probability of breeding across species
            'survival_rate': 0.2, # Percentage of best genomes to survive per species
            'stagnation_threshold': 15, # Generations a species can go without improvement before extinction
            'min_species_size': 2, # Minimum number of genomes in a species
            'elitism_species_count': 1
        }
        if config:
            self.config.update(config)

        self.populations = [] # List of lists, each inner list is a population
        self.global_generation = 0 # Tracks generations across all populations
        
        # Initialize all_time_best_fitness_per_population ONLY if num_populations is provided
        if self.num_populations is not None:
            self.all_time_best_fitness_per_population = [(-float('inf'))] * self.num_populations
        else:
            # If num_populations is None (e.g., when loading state), it will be set later.
            self.all_time_best_fitness_per_population = [] 

        # For adaptive mutation rates
        self.best_overall_fitness_history = []
        self.generations_since_last_improvement = 0

        if initialize_populations:
            self._initialize_multi_populations()

    def _initialize_multi_populations(self):
        """Initializes multiple distinct populations, each from a prototype."""
        for _ in range(self.num_populations):
            # Create a prototype for this population
            prototype_nn = NeuralNetwork(self.input_size, self.output_size, initialize_nodes=True)
            # Apply initial random mutations to the prototype to give it some initial diversity
            for _ in range(3): # Apply 3 initial mutations to the prototype
                prototype_nn._mutate_initial()

            current_population = []
            for _ in range(self.agents_per_population):
                agent = prototype_nn.clone()
                # Apply initial random mutations to each agent from its prototype
                for _ in range(3):
                    agent._mutate_initial()
                current_population.append(agent)
            self.populations.append(current_population)

    def _distance(self, genome1, genome2):
        """
        Calculates the genetic distance (compatibility) between two genomes.
        Formula: delta = c1 * E / N + c2 * D / N + c3 * W
        E = Excess genes, D = Disjoint genes, W = Average weight difference of matching genes.
        N = Number of genes in the larger genome (or 1 if N < 20).
        """
        nodes1, connections1 = genome1.get_genes()
        nodes2, connections2 = genome2.get_genes()

        # Convert connections to dictionaries for easier lookup by innovation number
        conns1_dict = {c.innovation_number: c for c in connections1}
        conns2_dict = {c.innovation_number: c for c in connections2}

        innovation_numbers1 = set(conns1_dict.keys())
        innovation_numbers2 = set(conns2_dict.keys())

        all_innovation_numbers = sorted(list(innovation_numbers1.union(innovation_numbers2)))

        excess_genes = 0
        disjoint_genes = 0
        matching_genes_weight_diff_sum = 0.0
        matching_genes_count = 0

        max_innovation1 = max(innovation_numbers1) if innovation_numbers1 else -1
        max_innovation2 = max(innovation_numbers2) if innovation_numbers2 else -1

        for inn in all_innovation_numbers:
            in_genome1 = inn in innovation_numbers1
            in_genome2 = inn in innovation_numbers2

            if in_genome1 and in_genome2:
                # Matching gene
                matching_genes_weight_diff_sum += abs(conns1_dict[inn].weight - conns2_dict[inn].weight)
                matching_genes_count += 1
            elif in_genome1 and inn > max_innovation2:
                # Excess gene (in genome1 but beyond max innovation of genome2)
                excess_genes += 1
            elif in_genome2 and inn > max_innovation1:
                # Excess gene (in genome2 but beyond max innovation of genome1)
                excess_genes += 1
            else:
                # Disjoint gene (present in one but not the other, and not excess)
                disjoint_genes += 1

        N = max(len(connections1), len(connections2))
        if N < 20: # NEAT paper suggests N=1 for small genomes
            N = 1

        # Avoid division by zero for matching_genes_count
        avg_weight_diff = matching_genes_weight_diff_sum / matching_genes_count if matching_genes_count > 0 else 0.0

        distance = (self.config['c1'] * excess_genes / N +
                    self.config['c2'] * disjoint_genes / N +
                    self.config['c3'] * avg_weight_diff)
        return distance

    def calculate_genetic_drift(self, genome1, genome2):
        """
        Calculates genetic drift (compatibility distance) between two genomes.
        This is essentially a wrapper for the _distance method.
        """
        return self._distance(genome1, genome2)

    def _speciate_single_population(self, population):
        """
        Divides a single population into species based on genetic compatibility.
        Returns the list of species for this population.
        """
        # First, reset members of existing species for the new generation
        # self.species is not initialized in GA, so it will be an empty list initially.
        # It's populated by Species objects in this method.
        if not hasattr(self, 'species'):
            self.species = []

        for species_obj in self.species: # self.species now holds Species objects
            species_obj.reset_members()

        # Try to add genomes to existing species
        for genome in population:
            found_species = False
            for species_obj in self.species:
                # The representative is the first genome added to the species in its creation
                if self._distance(genome, species_obj.representative) < self.config['compatibility_threshold']:
                    species_obj.add_member(genome)
                    found_species = True
                    break
            if not found_species:
                # Create a new species with this genome as the representative
                self.species.append(Species(genome))

        # Filter out empty species
        self.species = [s for s in self.species if s.members]

        # Update stagnation for all remaining species
        for species_obj in self.species:
            species_obj.update_stagnation()

        # Remove stagnant species (except the very best one globally, if it exists)
        # Find the best overall genome to determine the best species
        overall_best_genome = None
        overall_max_fitness = -float('inf')
        for species_obj in self.species:
            if species_obj.members:
                best_in_species = max(species_obj.members, key=lambda g: g.fitness)
                if best_in_species.fitness > overall_max_fitness:
                    overall_max_fitness = best_in_species.fitness
                    overall_best_genome = best_in_species

        surviving_species = []
        best_species_preserved = False
        for species_obj in self.species:
            # Preserve the species containing the overall best genome
            if overall_best_genome and any(g is overall_best_genome for g in species_obj.members):
                surviving_species.append(species_obj)
                best_species_preserved = True
            elif species_obj.generations_stagnant < self.config['stagnation_threshold']:
                surviving_species.append(species_obj)
            else:
                print(f"Species {species_obj.representative.node_id_counter} removed due to stagnation ({species_obj.generations_stagnant} generations).")
        
        self.species = surviving_species

        # Sort species by the fitness of their best member (for elitism/survival)
        if self.species: # Ensure species_list is not empty before sorting
            self.species.sort(key=lambda s: s.best_fitness_in_history, reverse=True)

        return self.species # Return the list of Species objects

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two parent genomes to create a child genome.
        Assumes parent1 is the fitter parent (or equally fit).
        """
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1 # Ensure parent1 is the fitter one

        child = NeuralNetwork(self.input_size, self.output_size, initialize_nodes=False) # Start with an empty NN
        child.nodes = {}
        child.connections = {}

        # 1. Inherit all input and output nodes from the fitter parent (parent1).
        # These are structural and must always be present.
        for node_id, node_gene in parent1.nodes.items():
            if node_gene.node_type in ['INPUT', 'OUTPUT']:
                child.nodes[node_id] = node_gene.clone()
        
        # 2. Inherit connections and associated hidden nodes.
        conns1_dict = {c.innovation_number: c for c in parent1.connections.values()}
        conns2_dict = {c.innovation_number: c for c in parent2.connections.values()}
        
        all_innovation_numbers = set(conns1_dict.keys()).union(set(conns2_dict.keys()))
        for inn in all_innovation_numbers:
            conn1 = conns1_dict.get(inn)
            conn2 = conns2_dict.get(inn)

            chosen_conn_gene = None
            if conn1 and conn2:
                # Matching connection: inherit randomly
                chosen_conn_gene = random.choice([conn1, conn2])
                # If one is disabled and the other enabled, child has 75% chance of being enabled
                if not conn1.enabled or not conn2.enabled:
                    if random.random() < 0.75:
                        chosen_conn_gene = ConnectionGene(chosen_conn_gene.from_node_id, chosen_conn_gene.to_node_id,
                                                          chosen_conn_gene.weight, True, chosen_conn_gene.innovation_number)
                    else:
                        chosen_conn_gene = ConnectionGene(chosen_conn_gene.from_node_id, chosen_conn_gene.to_node_id,
                                                          chosen_conn_gene.weight, False, chosen_conn_gene.innovation_number)
            elif conn1:
                # Disjoint/Excess from fitter parent (parent1)
                chosen_conn_gene = conn1
            elif conn2 and parent1.fitness == parent2.fitness:
                # Disjoint/Excess from less fit parent (parent2) if fitness is equal
                chosen_conn_gene = conn2
            
            if chosen_conn_gene:
                child.connections[inn] = chosen_conn_gene.clone() # Add a clone of the chosen connection

                # Ensure hidden nodes connected by this gene are added to the child
                for node_id in [chosen_conn_gene.from_node_id, chosen_conn_gene.to_node_id]:
                    if node_id not in child.nodes: # Only add if not already an input/output node
                        # Check if it's a hidden node in either parent
                        if parent1.nodes.get(node_id) and parent1.nodes[node_id].node_type == 'HIDDEN':
                            child.nodes[node_id] = parent1.nodes[node_id].clone()
                        elif parent2.nodes.get(node_id) and parent2.nodes[node_id].node_type == 'HIDDEN':
                            # If only in parent2, and parent1.fitness == parent2.fitness, or random chance
                            if parent1.fitness == parent2.fitness or random.random() < 0.5: # Apply some chance for less fit parent's hidden nodes
                                child.nodes[node_id] = parent2.nodes[node_id].clone()

        # Update child's node_id_counter based on all nodes
        child.node_id_counter = max(node.node_id for node in child.nodes.values()) + 1 if child.nodes else 0
        
        return child

    def _apply_random_mutation(self, genome):
        """Applies various mutations to a genome."""
        # Weight mutation
        if genome.connections:
            for conn in genome.connections.values():
                if random.random() < self.config['weight_mutate_prob']:
                    if random.random() < self.config['weight_replace_prob']:
                        conn.weight = random.uniform(-1.0, 1.0) # Replace weight
                    else:
                        conn.weight += random.gauss(0, 0.5) # Perturb weight
                    conn.weight = max(-5.0, min(5.0, conn.weight)) # Clamp weights

        # Add node mutation
        if random.random() < self.config['add_node_prob']:
            enabled_conns = [c for c in genome.connections.values() if c.enabled]
            if enabled_conns:
                conn_to_split = random.choice(enabled_conns)
                genome.add_node(conn_to_split.innovation_number)

        # Add connection mutation
        if random.random() < self.config['add_connection_prob']:
            # Try to add a new connection between two random nodes
            all_node_ids = list(genome.nodes.keys())
            if len(all_node_ids) >= 2:
                from_node_id = random.choice(all_node_ids)
                to_node_id = random.choice(all_node_ids)
                genome.add_connection(from_node_id, to_node_id)

        # Enable/Disable connection mutation
        if random.random() < self.config['enable_disable_prob']:
            if genome.connections:
                conn = random.choice(list(genome.connections.values()))
                conn.enabled = not conn.enabled

    def get_new_random_genome(self):
        """Creates and returns a new, randomly initialized NeuralNetwork."""
        new_genome = NeuralNetwork(self.input_size, self.output_size, initialize_nodes=True)
        for _ in range(3): # Apply initial mutations to give it some diversity
            new_genome._mutate_initial()
        return new_genome


    def evolve(self, fitness_evaluator_func): # Now accepts fitness_evaluator_func
        """
        Evolves all populations for one generation, including inter-population replacement.
        Args:
            fitness_evaluator_func (callable): A function that takes a genome and returns its fitness.
                                               Used for evaluating mutations.
        """
        self.global_generation += 1
        
        # Step 1: Evolve each individual population
        for i, population in enumerate(self.populations):
            # Speciate this population
            # self.species is a list of Species objects for the current population
            species_for_this_population = self._speciate_single_population(population)

            new_sub_population = []

            # Calculate total adjusted fitness for proportional selection within this population
            # Use the best fitness of each species for offspring allocation
            total_adjusted_fitness = sum(s.best_fitness_in_history for s in species_for_this_population if s.members)

            # Handle case where no species or all adjusted fitnesses are zero in this population
            if total_adjusted_fitness <= 0 and species_for_this_population:
                # If all fitnesses are non-positive, distribute offspring equally
                offspring_counts = {id(s): self.agents_per_population // len(species_for_this_population) for s in species_for_this_population}
                for j in range(self.agents_per_population % len(species_for_this_population)):
                    offspring_counts[id(species_for_this_population[j])] += 1
            elif not species_for_this_population:
                # If a population somehow becomes empty of species, re-initialize it
                print(f"Warning: Population {i} has no species. Re-initializing.")
                new_random_pop = []
                # Create a new prototype for this re-initialized population
                prototype_nn = NeuralNetwork(self.input_size, self.output_size, initialize_nodes=True)
                for _ in range(3): prototype_nn._mutate_initial() # Apply initial mutations to the prototype
                
                for _ in range(self.agents_per_population):
                    agent = prototype_nn.clone()
                    for _ in range(3): agent._mutate_initial() # Apply initial mutations to each agent
                    new_random_pop.append(agent)
                self.populations[i] = new_random_pop
                # Update all-time best for this re-initialized population
                # Ensure the list is large enough for the index 'i'
                if i >= len(self.all_time_best_fitness_per_population):
                    # This should ideally not happen if num_populations is consistent,
                    # but as a safeguard, extend the list.
                    self.all_time_best_fitness_per_population.extend([-float('inf')] * (i + 1 - len(self.all_time_best_fitness_per_population)))
                self.all_time_best_fitness_per_population[i] = max(self.all_time_best_fitness_per_population[i], max(g.fitness for g in new_random_pop))
                continue # Skip to next population
            else:
                offspring_counts = {}
                for species_obj in species_for_this_population:
                    if not species_obj.members: continue
                    count = int(round(species_obj.best_fitness_in_history / total_adjusted_fitness * self.agents_per_population))
                    offspring_counts[id(species_obj)] = count

            current_offspring_sum = sum(offspring_counts.values())
            if current_offspring_sum < self.agents_per_population:
                remaining = self.agents_per_population - current_offspring_sum
                for _ in range(remaining):
                    if species_for_this_population: # Ensure there's at least one species
                        # Give remaining slots to the best species (first in sorted list)
                        best_species_id = id(species_for_this_population[0])
                        offspring_counts[best_species_id] = offspring_counts.get(best_species_id, 0) + 1
            elif current_offspring_sum > self.agents_per_population:
                excess = current_offspring_sum - self.agents_per_population
                # Remove from worst species first (last in sorted list, or those with fewest offspring)
                sorted_species_by_offspring = sorted(species_for_this_population, key=lambda s: offspring_counts.get(id(s), 0))
                for _ in range(excess):
                    for s in reversed(sorted_species_by_offspring):
                        if offspring_counts.get(id(s), 0) > 0:
                            offspring_counts[id(s)] -= 1
                            break

            for species_obj in species_for_this_population:
                species_obj.members.sort(key=lambda g: g.fitness, reverse=True)
                num_survivors = max(1, int(len(species_obj.members) * self.config['survival_rate']))
                for k in range(min(num_survivors, len(species_obj.members))):
                    new_sub_population.append(species_obj.members[k].clone())

                current_species_id = id(species_obj)
                num_offspring_to_create = offspring_counts.get(current_species_id, 0) - num_survivors
                if num_offspring_to_create < 0: num_offspring_to_create = 0

                for _ in range(num_offspring_to_create):
                    parent1 = random.choice(species_obj.members[:num_survivors])
                    parent2 = random.choice(species_obj.members[:num_survivors])
                    child = self._crossover(parent1, parent2)
                    
                    # --- Gradient Descent-like Mutation (Trial and Revert) ---
                    # Evaluate child's fitness before mutation.
                    # We clone here to ensure the evaluation doesn't modify the child before mutation.
                    original_child_genome_clone = child.clone()
                    original_child_fitness = fitness_evaluator_func(original_child_genome_clone)

                    # Apply mutation to the actual child genome
                    self._apply_random_mutation(child)

                    # Evaluate mutated child's fitness
                    mutated_child_fitness = fitness_evaluator_func(child)

                    # Decide whether to keep or revert the mutation
                    if mutated_child_fitness < original_child_fitness:
                        # Revert: restore original structure and fitness
                        child.set_genes(*original_child_genome_clone.get_genes())
                        child.fitness = original_child_fitness
                        # print(f"Mutation reverted. Original: {original_child_fitness:.2f}, Mutated: {mutated_child_fitness:.2f}")
                    else:
                        # Keep: update fitness to the improved/neutral value
                        child.fitness = mutated_child_fitness
                        # print(f"Mutation kept. Original: {original_child_fitness:.2f}, Mutated: {mutated_child_fitness:.2f}")
                    # --- End Gradient Descent-like Mutation ---

                    new_sub_population.append(child)

            while len(new_sub_population) < self.agents_per_population:
                new_sub_population.append(NeuralNetwork(self.input_size, self.output_size, initialize_nodes=True))

            self.populations[i] = new_sub_population[:self.agents_per_population]
            # Update all-time best fitness for this population after evolution
            if self.populations[i]:
                current_pop_best_fitness = max(g.fitness for g in self.populations[i])
                # Ensure the list is large enough for the index 'i'
                if i >= len(self.all_time_best_fitness_per_population):
                    self.all_time_best_fitness_per_population.extend([-float('inf')] * (i + 1 - len(self.all_time_best_fitness_per_population)))
                self.all_time_best_fitness_per_population[i] = max(self.all_time_best_fitness_per_population[i], current_pop_best_fitness)


        # Find the overall best genome across all populations for adaptive mutation rates
        overall_best_genome_this_gen = None
        overall_max_fitness_this_gen = -float('inf')
        for population in self.populations:
            if population:
                best_in_pop = max(population, key=lambda g: g.fitness)
                if best_in_pop.fitness > overall_max_fitness_this_gen:
                    overall_max_fitness_this_gen = best_in_pop.fitness
                    overall_best_genome_this_gen = best_in_pop.clone()

        # Adaptive Mutation Rates Logic
        if self.best_overall_fitness_history and overall_max_fitness_this_gen > self.best_overall_fitness_history[-1]:
            self.best_overall_fitness_history.append(overall_max_fitness_this_gen)
            self.generations_since_last_improvement = 0
            # Slightly decrease mutation rates
            self.config['add_node_prob'] = max(0.01, self.config['add_node_prob'] * 0.95)
            self.config['add_connection_prob'] = max(0.01, self.config['add_connection_prob'] * 0.95)
            self.config['weight_mutate_prob'] = max(0.5, self.config['weight_mutate_prob'] * 0.98)
            print("Mutation rates decreased due to fitness improvement.")
        else:
            self.generations_since_last_improvement += 1
            if self.generations_since_last_improvement > self.config['stagnation_threshold']:
                # Slightly increase mutation rates if stagnating
                self.config['add_node_prob'] = min(0.1, self.config['add_node_prob'] * 1.1)
                self.config['add_connection_prob'] = min(0.15, self.config['add_connection_prob'] * 1.1)
                self.config['weight_mutate_prob'] = min(0.95, self.config['weight_mutate_prob'] * 1.02)
                print(f"Mutation rates increased due to stagnation ({self.generations_since_last_improvement} generations).")
            if not self.best_overall_fitness_history: # For the very first generation
                self.best_overall_fitness_history.append(overall_max_fitness_this_gen)


        # Step 2: Periodic population replacement
        if self.global_generation % 50 == 0: # Changed from 10 to 50
            print(f"--- Global Generation {self.global_generation}: Performing population replacement ---")
            # Get best fitness for each population
            population_scores = []
            for i, pop in enumerate(self.populations):
                if pop:
                    population_scores.append((max(g.fitness for g in pop), i))
                else:
                    population_scores.append((-float('inf'), i)) # Handle empty populations

            # Sort populations by their best fitness
            population_scores.sort(key=lambda x: x[0], reverse=True)

            # Identify the two worst populations (indices)
            worst_pop_indices = [score[1] for score in population_scores[-2:]]
            
            # Identify the top 3 populations (indices)
            top_pop_indices = [score[1] for score in population_scores[:3]]

            # Select elite genomes from the top 3 populations
            elite_genomes = []
            for idx in top_pop_indices:
                top_pop = self.populations[idx]
                # Take the top N% of genomes from each top population for elitist breeding
                num_elites_from_pop = max(1, int(len(top_pop) * 0.1)) # Take top 10%
                top_pop.sort(key=lambda g: g.fitness, reverse=True)
                elite_genomes.extend(top_pop[:num_elites_from_pop])
            
            # Ensure we have at least two elites to breed from
            if len(elite_genomes) < 2:
                print("Warning: Not enough elite genomes for breeding. Re-initializing replaced populations randomly.")
                for idx in worst_pop_indices:
                    new_pop = []
                    for _ in range(self.agents_per_population):
                        agent = NeuralNetwork(self.input_size, self.output_size, initialize_nodes=True)
                        for _ in range(3): agent._mutate_initial()
                        new_pop.append(agent)
                    self.populations[idx] = new_pop
                    # Update all-time best for this re-initialized population
                    # Ensure the list is large enough for the index 'idx'
                    if idx >= len(self.all_time_best_fitness_per_population):
                        self.all_time_best_fitness_per_population.extend([-float('inf')] * (idx + 1 - len(self.all_time_best_fitness_per_population)))
                    self.all_time_best_fitness_per_population[idx] = max(self.all_time_best_fitness_per_population[idx], max(g.fitness for g in new_pop))
            else:
                # Create new prototype genomes by breeding from elites
                new_prototypes = []
                # Create 2 new prototypes for the 2 replaced populations
                for _ in range(len(worst_pop_indices)):
                    parent1 = random.choice(elite_genomes)
                    parent2 = random.choice(elite_genomes)
                    # Ensure parents are distinct if possible
                    while parent1 is parent2 and len(elite_genomes) > 1:
                        parent2 = random.choice(elite_genomes)
                    
                    new_prototype = self._crossover(parent1, parent2)
                    # Apply some mutation to the new prototype to ensure diversity
                    self._apply_random_mutation(new_prototype) # Use the new mutation method
                    new_prototypes.append(new_prototype)

                # Replace the two worst populations
                for i, idx in enumerate(worst_pop_indices):
                    new_pop = []
                    prototype_for_new_pop = new_prototypes[i]
                    for _ in range(self.agents_per_population):
                        agent = prototype_for_new_pop.clone()
                        # Apply initial mutations to agents from the new prototype
                        for _ in range(3):
                            agent._mutate_initial()
                        new_pop.append(agent)
                    self.populations[idx] = new_pop
                    # Update all-time best for this re-initialized population
                    # Ensure the list is large enough for the index 'idx'
                    if idx >= len(self.all_time_best_fitness_per_population):
                        self.all_time_best_fitness_per_population.extend([-float('inf')] * (idx + 1 - len(self.all_time_best_fitness_per_population)))
                    self.all_time_best_fitness_per_population[idx] = max(self.all_time_best_fitness_per_population[idx], max(g.fitness for g in new_pop))
                print(f"Replaced populations {worst_pop_indices} with new ones from elite breeding.")

        # Find the overall best genome across all populations for display
        overall_best_genome = None
        overall_max_fitness = -float('inf')
        for population in self.populations:
            if population:
                best_in_pop = max(population, key=lambda g: g.fitness)
                if best_in_pop.fitness > overall_max_fitness:
                    overall_max_fitness = best_in_pop.fitness
                    overall_best_genome = best_in_pop.clone()

        print(f"Global Generation {self.global_generation} complete. Overall best fitness: {overall_max_fitness:.4f}")
        print(f"Number of active populations: {len(self.populations)}") # Should always be 5

        return overall_best_genome


# --- Pygame Simulation Classes ---

# Constants for the simulation
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 700
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (100, 100, 100)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (200, 200, 200)
DARK_GRAY = (50, 50, 50)
ORANGE = (255, 165, 0) # Added for thruster detail
FUEL_GAUGE_COLOR_FULL = (0, 255, 0) # Green for full
FUEL_GAUGE_COLOR_EMPTY = (150, 0, 0) # Dark red for empty

# Lander physics constants
GRAVITY = 0.0005 # Pixels per ms^2, adjusted for Pygame's typical scale
MAIN_THRUST_POWER = 0.002
# SIDE_THRUST_POWER = 0.00005 # Removed
ANGULAR_DRAG = 0.95 # Increased angular drag (closer to 0 for more drag)
LINEAR_DRAG = 0.999 # Multiplier for linear velocity each frame

INITIAL_FUEL = 30000 # ~30 seconds at 1 unit/ms main thrust consumption
FUEL_CONSUMPTION_MAIN = 1.0 # Units per ms
# FUEL_CONSUMPTION_SIDE = 0.1 # Removed

MAX_SIMULATION_TIME_MS = 45000 # Increased from 30000 (30 seconds to 45 seconds)

# Landing criteria
MAX_LANDING_VX = 0.05
MAX_LANDING_VY = 0.1
MAX_LANDING_ANGLE_RAD = math.radians(5) # +/- 5 degrees for soft landing

# New constant for maximum flight angle before crashing
MAX_FLIGHT_ANGLE_RAD = math.radians(90) # +/- 90 degrees for crash during flight

# New constant for limiting angular acceleration (now from main engine angling)
# 5 degrees per second squared, converted to radians per millisecond squared
MAX_ANGULAR_ACCELERATION_RAD_PER_MS2 = math.radians(5) / (1000.0 * 1000.0)

# New Fitness Constants for Crash Penalties
OUT_OF_BOUNDS_CRASH_PENALTY = -200.0 # Most severe penalty
ANGLE_CRASH_PENALTY = -150.0       # Severe penalty for tipping over
GROUND_CRASH_BASE_PENALTY = -50.0  # Base penalty for ground crash (even if low velocity)
GROUND_CRASH_VELOCITY_PENALTY_MAX = 200.0 # Max additional penalty for high velocity ground crash
CRITICAL_IMPACT_SPEED = 0.8        # Velocity magnitude at which full velocity penalty is applied
MIN_FITNESS_CAP_ON_CRASH = -1000.0 # Cap the minimum fitness an agent can get after a crash

# New fitness constants for continuous rewards/penalties
PROXIMITY_OVER_TIME_REWARD_COEFF = 0.001 # Small reward per ms for proximity
VELOCITY_CONTROL_REWARD_COEFF = 0.5 # Reward for low velocities while flying
ANGULAR_VELOCITY_CONTROL_REWARD_COEFF = 0.2 # Reward for low angular velocity while flying
FUEL_WASTE_PENALTY_COEFF = 0.0001 # Penalty for thrusting while falling fast

# Max expected flight velocities for normalization of rewards
MAX_FLIGHT_VX = 0.5 # Example max horizontal velocity
MAX_FLIGHT_VY = 0.5 # Example max vertical velocity (can be higher for falling)
MAX_FLIGHT_ANGULAR_VELOCITY_RAD = math.radians(10) # Example max angular velocity


class Lander:
    def __init__(self, start_x, start_y, initial_fuel=INITIAL_FUEL):
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = random.uniform(-math.pi/4, math.pi/4) # Radians, 0 is straight up
        self.angular_velocity = 0.0
        self.fuel = initial_fuel
        self.initial_fuel = initial_fuel
        self.mass = 1.0 # Base mass
        self.base_mass = 1.0
        self.fuel_mass_ratio = 0.00005 # How much fuel affects mass (e.g., 1 unit of fuel is 0.00005 mass units)

        self.width = 20
        self.height = 30
        self.color = WHITE # Default color

        # No more thrust_left_on, thrust_right_on
        self.thrust_main_power_input = 0.0 # NN output 0-1
        self.angular_velocity_target_input = 0.0 # NN output -1 to 1

        self.landed = False
        self.crashed = False
        self.score = 0.0 # This is the raw fitness score
        self.time_elapsed_ms = 0
        self.crash_reason = None # New: Stores why the lander crashed

        # New for continuous fitness feedback
        self.cumulative_proximity_score = 0.0

    def update(self, dt):
        if self.landed or self.crashed:
            return

        self.time_elapsed_ms += dt

        # Update mass based on fuel
        self.mass = self.base_mass + (self.fuel * self.fuel_mass_ratio)

        # Apply main thrust based on NN output
        force_x = 0.0
        force_y = 0.0
        
        main_thrust_magnitude = self.thrust_main_power_input * MAIN_THRUST_POWER
        if main_thrust_magnitude > 0 and self.fuel > 0:
            force_x += main_thrust_magnitude * math.sin(self.angle) * dt
            force_y -= main_thrust_magnitude * math.cos(self.angle) * dt # Thrust acts upwards relative to lander
            self.fuel = max(0, self.fuel - (main_thrust_magnitude / MAIN_THRUST_POWER) * FUEL_CONSUMPTION_MAIN * dt) # Consume fuel proportional to thrust applied

        # Apply gravity
        force_y += GRAVITY * dt

        # Update velocities
        self.vx += force_x / self.mass
        self.vy += force_y / self.mass

        # Apply linear drag
        self.vx *= LINEAR_DRAG
        self.vy *= LINEAR_DRAG

        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt

        # Angular control: Adjust angular velocity towards target
        # The NN's second output directly influences angular velocity change
        target_angular_accel = self.angular_velocity_target_input * MAX_ANGULAR_ACCELERATION_RAD_PER_MS2
        self.angular_velocity += target_angular_accel * dt
        
        # Apply angular drag
        self.angular_velocity *= ANGULAR_DRAG
        
        # Update angle
        self.angle += self.angular_velocity * dt

        # Keep angle within -pi to pi range
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        # Check fuel
        if self.fuel <= 0:
            self.fuel = 0
            self.thrust_main_power_input = 0.0 # No thrust if no fuel

    def get_corners(self):
        """Returns the coordinates of the lander's four corners."""
        half_width = self.width / 2
        half_height = self.height / 2

        # Lander's local coordinates relative to its center
        points = [
            (-half_width, -half_height), # Top-left
            (half_width, -half_height),  # Top-right
            (half_width, half_height),   # Bottom-right
            (-half_width, half_height)   # Bottom-left
        ]

        rotated_points = []
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        for px, py in points:
            # Rotate
            rx = px * cos_angle - py * sin_angle
            ry = px * sin_angle + py * cos_angle
            # Translate to world coordinates
            rotated_points.append((self.x + rx, self.y + ry))
        return rotated_points

    def draw(self, screen):
        # Calculate rotation and translation for all parts
        cos_angle = math.cos(self.angle)
        sin_angle = math.sin(self.angle)

        def rotate_and_translate(px, py):
            rx = px * cos_angle - py * sin_angle
            ry = px * sin_angle + py * cos_angle
            return self.x + rx, self.y + ry

        # Main body (a slightly tapered rectangle)
        body_points = [
            rotate_and_translate(-self.width * 0.4, -self.height * 0.5), # Top-left (tapered)
            rotate_and_translate(self.width * 0.4, -self.height * 0.5),  # Top-right (tapered)
            rotate_and_translate(self.width * 0.5, self.height * 0.5),   # Bottom-right
            rotate_and_translate(-self.width * 0.5, self.height * 0.5)   # Bottom-left
        ]
        pygame.draw.polygon(screen, self.color, body_points)
        pygame.draw.polygon(screen, DARK_GRAY, body_points, 1) # Border

        # Nose cone (triangle on top)
        nose_points = [
            rotate_and_translate(0, -self.height * 0.65), # Tip
            rotate_and_translate(-self.width * 0.4, -self.height * 0.5), # Base left
            rotate_and_translate(self.width * 0.4, -self.height * 0.5)  # Base right
        ]
        pygame.draw.polygon(screen, LIGHT_GRAY, nose_points)
        pygame.draw.polygon(screen, DARK_GRAY, nose_points, 1) # Border

        # Cockpit window (small rectangle or circle)
        window_center_x = 0
        window_center_y = -self.height * 0.3
        window_radius = self.width * 0.2
        window_pos = rotate_and_translate(window_center_x, window_center_y)
        pygame.draw.circle(screen, BLUE, (int(window_pos[0]), int(window_pos[1])), int(window_radius))
        pygame.draw.circle(screen, DARK_GRAY, (int(window_pos[0]), int(window_pos[1])), int(window_radius), 1)

        # Landing Legs (lines extending from bottom corners)
        leg_length = self.height * 0.4
        leg_offset_x = self.width * 0.4 # Offset from center for leg attachment

        # Left Leg
        leg_left_start = rotate_and_translate(-leg_offset_x, self.height * 0.5)
        leg_left_end = rotate_and_translate(-leg_offset_x - self.width * 0.2, self.height * 0.5 + leg_length)
        pygame.draw.line(screen, GRAY, leg_left_start, leg_left_end, 2)
        # Foot pad for left leg
        foot_left_start = rotate_and_translate(-leg_offset_x - self.width * 0.2 - self.width * 0.1, self.height * 0.5 + leg_length)
        foot_left_end = rotate_and_translate(-leg_offset_x - self.width * 0.2 + self.width * 0.1, self.height * 0.5 + leg_length)
        pygame.draw.line(screen, GRAY, foot_left_start, foot_left_end, 3)


        # Right Leg
        leg_right_start = rotate_and_translate(leg_offset_x, self.height * 0.5)
        leg_right_end = rotate_and_translate(leg_offset_x + self.width * 0.2, self.height * 0.5 + leg_length)
        pygame.draw.line(screen, GRAY, leg_right_start, leg_right_end, 2)
        # Foot pad for right leg
        foot_right_start = rotate_and_translate(leg_offset_x + self.width * 0.2 - self.width * 0.1, self.height * 0.5 + leg_length)
        foot_right_end = rotate_and_translate(leg_offset_x + self.width * 0.2 + self.width * 0.1, self.height * 0.5 + leg_length)
        pygame.draw.line(screen, GRAY, foot_right_start, foot_right_end, 3)


        # Draw main thruster flame if main thrust is on
        if self.thrust_main_power_input > 0.1 and self.fuel > 0: # Only draw if significant thrust
            flame_length = 10 + random.randint(0, 5) * self.thrust_main_power_input # Flicker and scale with thrust
            flame_width = self.width / 2 * self.thrust_main_power_input

            # Calculate the true bottom center of the rotated lander
            # Local bottom center is (0, self.height / 2) relative to lander's center (self.x, self.y)
            # Rotate this local point and add to global position
            rotated_bottom_x_offset = 0 * cos_angle - (self.height / 2) * sin_angle
            rotated_bottom_y_offset = 0 * sin_angle + (self.height / 2) * cos_angle

            bottom_center_x = self.x + rotated_bottom_x_offset
            bottom_center_y = self.y + rotated_bottom_y_offset

            # Flame extends in the direction of the thrust (opposite to lander's nose)
            # Lander's nose direction is (sin(angle), cos(angle)) if 0 is up.
            # Thrust is in (-sin(angle), -cos(angle)) direction.
            # Flame should be in the direction of thrust.
            # Corrected: Flame points in the direction of thrust.
            flame_tip_x = bottom_center_x - flame_length * math.sin(self.angle)
            flame_tip_y = bottom_center_y + flame_length * math.cos(self.angle) # Corrected sign here for downward flame

            # Base of flame should be perpendicular to the flame direction
            # Perpendicular to (-sin(angle), -cos(angle)) are (cos(angle), -sin(angle)) and (-cos(angle), sin(angle))
            flame_base_left_x = bottom_center_x + flame_width / 2 * math.cos(self.angle)
            flame_base_left_y = bottom_center_y - flame_width / 2 * math.sin(self.angle)

            flame_base_right_x = bottom_center_x - flame_width / 2 * math.cos(self.angle)
            flame_base_right_y = bottom_center_y + flame_width / 2 * math.sin(self.angle)

            pygame.draw.polygon(screen, YELLOW, [(flame_tip_x, flame_tip_y),
                                                 (flame_base_left_x, flame_base_left_y),
                                                 (flame_base_right_x, flame_base_right_y)])

        # Draw Fuel Gauge
        gauge_width = self.width * 0.3 # Increased width
        gauge_height = self.height * 0.5 # Increased height
        
        # New offset calculations for centering and aligning with bottom edge
        gauge_offset_x = 0 # Centered horizontally
        gauge_offset_y = self.height * 0.5 - gauge_height / 2 # Aligned with bottom edge of main body

        # Background for the gauge (empty part)
        gauge_bg_points = [
            rotate_and_translate(gauge_offset_x - gauge_width / 2, gauge_offset_y - gauge_height / 2),
            rotate_and_translate(gauge_offset_x + gauge_width / 2, gauge_offset_y - gauge_height / 2),
            rotate_and_translate(gauge_offset_x + gauge_width / 2, gauge_offset_y + gauge_height / 2),
            rotate_and_translate(gauge_offset_x - gauge_width / 2, gauge_offset_y + gauge_height / 2)
        ]
        pygame.draw.polygon(screen, LIGHT_GRAY, gauge_bg_points, 0) # Filled background with LIGHT_GRAY
        pygame.draw.polygon(screen, BLACK, gauge_bg_points, 2) # Thicker border

        # Filled part of the gauge
        fuel_ratio = self.fuel / self.initial_fuel
        current_fuel_height = gauge_height * fuel_ratio
        
        # Calculate color based on fuel level (green to red)
        r = int(255 * (1 - fuel_ratio))
        g = int(255 * fuel_ratio)
        fuel_color = (r, g, 0) # Interpolate between red and green

        fuel_fill_points = [
            rotate_and_translate(gauge_offset_x - gauge_width / 2, gauge_offset_y + gauge_height / 2), # Bottom-left
            rotate_and_translate(gauge_offset_x + gauge_width / 2, gauge_offset_y + gauge_height / 2), # Bottom-right
            rotate_and_translate(gauge_offset_x + gauge_width / 2, gauge_offset_y + gauge_height / 2 - current_fuel_height), # Top-right (fuel level)
            rotate_and_translate(gauge_offset_x - gauge_width / 2, gauge_offset_y + gauge_height / 2 - current_fuel_height)  # Top-left (fuel level)
        ]
        pygame.draw.polygon(screen, fuel_color, fuel_fill_points, 0)


    def reset(self, start_x, start_y):
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = random.uniform(-math.pi/4, math.pi/4) # Radians, 0 is straight up
        self.angular_velocity = 0.0
        self.fuel = self.initial_fuel
        self.mass = self.base_mass + (self.fuel * self.fuel_mass_ratio)
        self.thrust_main_power_input = 0.0
        self.angular_velocity_target_input = 0.0
        self.landed = False
        self.crashed = False
        self.score = 0.0
        self.time_elapsed_ms = 0
        self.color = WHITE # Reset color on reset
        self.crash_reason = None # Reset crash reason
        self.cumulative_proximity_score = 0.0 # Reset cumulative score

class Terrain:
    def __init__(self, screen_width, screen_height, num_points=20, landing_zone_width=100):
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.num_points = num_points
        self.landing_zone_width = landing_zone_width
        self.points = [] # List of (x, y) coordinates
        self.landing_zone_x = 0 # X-coordinate of the start of the landing zone
        self.landing_zone_y = 0 # Y-coordinate of the landing zone

        self.generate_terrain()

    def generate_terrain(self):
        self.points = []
        
        # Determine landing zone properties
        min_landing_x = self.screen_width * 0.2
        max_landing_x = self.screen_width * 0.8 - self.landing_zone_width
        self.landing_zone_x = random.randint(int(min_landing_x), int(max_landing_x))
        self.landing_zone_y = random.randint(self.screen_height - 100, self.screen_height - 50)

        # 1. Generate points for the left side of the screen up to the landing zone start
        current_x = 0
        current_y = random.randint(self.screen_height // 2, self.screen_height - 50)
        self.points.append((current_x, current_y))

        while current_x < self.landing_zone_x:
            segment_length = random.randint(50, 150)
            next_x = current_x + segment_length
            
            # Ensure we don't overshoot the landing zone start
            if next_x >= self.landing_zone_x:
                next_x = self.landing_zone_x
                next_y = self.landing_zone_y # Force to landing zone height
            else:
                next_y = current_y + random.randint(-50, 50)
                next_y = max(self.screen_height // 2, min(self.screen_height - 50, next_y))

            self.points.append((next_x, next_y))
            current_x = next_x
            current_y = next_y

        # 2. Add the exact landing zone segment
        # Ensure the point at landing_zone_x is exactly at landing_zone_y
        # This handles cases where the last point generated before the loop ended was not exactly at landing_zone_x
        if self.points[-1][0] != self.landing_zone_x or self.points[-1][1] != self.landing_zone_y:
            self.points.append((self.landing_zone_x, self.landing_zone_y))
        
        # Add the end point of the landing zone
        self.points.append((self.landing_zone_x + self.landing_zone_width, self.landing_zone_y))
        current_x = self.landing_zone_x + self.landing_zone_width
        current_y = self.landing_zone_y # Start generating from this height

        # 3. Generate points for the right side of the screen
        while current_x < self.screen_width:
            segment_length = random.randint(50, 150)
            next_x = current_x + segment_length
            next_y = current_y + random.randint(-50, 50)
            next_y = max(self.screen_height // 2, min(self.screen_height - 50, next_y))
            self.points.append((next_x, next_y))
            current_x = next_x
            current_y = next_y

        # Ensure the last point reaches the right edge
        if self.points[-1][0] < self.screen_width:
            self.points.append((self.screen_width, self.points[-1][1]))

        # Add closing points for drawing the polygon
        self.points.append((self.screen_width, self.screen_height))
        self.points.append((0, self.screen_height))

    def get_height_at_x(self, x):
        """Finds the terrain height at a given x-coordinate using linear interpolation."""
        if x < self.points[0][0]: return self.points[0][1]
        if x > self.points[-3][0]: return self.points[-3][1] # -3 because last two points are for closing polygon

        for i in range(len(self.points) - 3): # Iterate through segments, excluding closing points
            p1 = self.points[i]
            p2 = self.points[i+1]
            if p1[0] <= x <= p2[0]:
                if p2[0] == p1[0]: # Vertical line, just return p1's y
                    return p1[1]
                # Linear interpolation
                return p1[1] + (p2[1] - p1[1]) * (x - p1[0]) / (p2[0] - p1[0])
        return self.screen_height # Should not happen if x is within bounds

    def get_slope_at_x(self, x):
        """Calculates the slope (angle) of the terrain at a given x-coordinate."""
        if x < self.points[0][0] or x > self.points[-3][0]: return 0.0 # Flat outside main terrain

        for i in range(len(self.points) - 3):
            p1 = self.points[i]
            p2 = self.points[i+1]
            if p1[0] <= x <= p2[0]:
                if p2[0] == p1[0]: return 0.0 # Vertical segment, treat as flat for slope
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                return math.atan(slope) # Returns angle in radians
        return 0.0

    def check_collision(self, lander):
        """Checks if the lander has collided with the terrain."""
        lander_corners = lander.get_corners()
        for cx, cy in lander_corners:
            if cy >= self.get_height_at_x(cx):
                return True # Collision detected
        return False

    def is_in_landing_zone(self, lander):
        """Checks if the lander's center is within the designated landing zone."""
        return self.landing_zone_x <= lander.x <= (self.landing_zone_x + self.landing_zone_width)

    def draw(self, screen):
        pygame.draw.polygon(screen, GRAY, self.points)
        pygame.draw.lines(screen, BLACK, False, self.points[:-2], 3) # Draw top line thicker

        # Draw landing zone marker
        pygame.draw.line(screen, GREEN,
                         (self.landing_zone_x, self.landing_zone_y),
                         (self.landing_zone_x + self.landing_zone_width, self.landing_zone_y),
                         5)


class LunarLanderSimulation:
    def __init__(self, screen_width, screen_height, neat_ga_instance, render_enabled=True):
        pygame.init()
        self.screen_width = screen_width
        self.screen_height = screen_height
        self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
        pygame.display.set_caption("NEAT Lunar Lander")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 24)
        self.render_enabled = render_enabled

        self.neat_ga = neat_ga_instance
        self.terrain = Terrain(self.screen_width, self.screen_height)
        # Store the successful landings count from the previous generation
        self.previous_successful_landings_count = 0 

        # Initial lander position (top center, slightly offset)
        self.lander_start_x = self.screen_width / 2
        self.lander_start_y = 50

        # Attributes to track state across agents within a generation for terrain randomization
        self.successful_landings_count_current_gen = 0
        self.terrain_randomized_this_gen = False
        self.current_generation_number = -1 # To detect start of new generation

        self.best_genome_for_display = None # Store the best genome for visualization
        self.show_nn_overlay = True # New: Toggle for NN overlay visibility

        # Colors for each population
        self.population_colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Yellow
            (255, 165, 0)   # Orange
        ]
        # Ensure enough colors for all populations
        if len(self.population_colors) < self.neat_ga.num_populations:
            # Add more random colors if needed (simple approach)
            for _ in range(self.neat_ga.num_populations - len(self.population_colors)):
                self.population_colors.append((random.randint(50, 200), random.randint(50, 200), random.randint(50, 200)))

        self.current_population_avg_fitnesses = [] # Stores (avg_fitness, pop_idx) for display


    def _get_lander_state_for_nn(self, lander):
        """
        Extracts and normalizes relevant state information for the neural network.
        Inputs: x, y, vx, vy, angle, angular_velocity, fuel,
                distance_to_landing_zone_x, distance_to_lz_y,
                terrain_height_at_lander_x, terrain_slope_at_lander_x,
                altitude_above_terrain (NEW)
        """
        # Normalize values to be between 0 and 1 or -1 and 1
        norm_x = lander.x / self.screen_width
        norm_y = lander.y / self.screen_height
        norm_vx = lander.vx / (MAX_FLIGHT_VX * 2) # Use MAX_FLIGHT_VX for normalization
        norm_vy = lander.vy / (MAX_FLIGHT_VY * 2) # Use MAX_FLIGHT_VY for normalization
        norm_angle = lander.angle / math.pi # -1 to 1 range for -pi to pi
        norm_angular_velocity = lander.angular_velocity / MAX_FLIGHT_ANGULAR_VELOCITY_RAD # Max angular velocity for normalization
        norm_fuel = lander.fuel / lander.initial_fuel

        # Distance to landing zone
        landing_zone_center_x = self.terrain.landing_zone_x + self.terrain.landing_zone_width / 2
        dist_to_lz_x = (landing_zone_center_x - lander.x) / self.screen_width
        dist_to_lz_y = (self.terrain.landing_zone_y - lander.y) / self.screen_height

        # Terrain info at lander's x
        terrain_height_at_lander = self.terrain.get_height_at_x(lander.x)
        norm_terrain_height = terrain_height_at_lander / self.screen_height
        terrain_slope_at_lander = self.terrain.get_slope_at_x(lander.x)
        norm_terrain_slope = terrain_slope_at_lander / (math.pi / 2) # Normalized to -1 to 1 for -pi/2 to pi/2

        # New: Altitude above terrain
        altitude_above_terrain = lander.y - terrain_height_at_lander
        # Normalize altitude: 0 at terrain, 1 at top of screen (or max reasonable altitude)
        norm_altitude = altitude_above_terrain / self.screen_height # Can be negative if below terrain
        
        return [
            norm_x, norm_y,
            norm_vx, norm_vy,
            norm_angle, norm_angular_velocity,
            norm_fuel,
            dist_to_lz_x, dist_to_lz_y,
            norm_terrain_height, norm_terrain_slope,
            norm_altitude # NEW INPUT
        ]

    def _is_soft_landing(self, lander):
        """Checks if the lander meets the velocity and angle criteria for a soft landing."""
        return abs(lander.vx) < MAX_LANDING_VX and \
               abs(lander.vy) < MAX_LANDING_VY and \
               abs(lander.angle) < MAX_LANDING_ANGLE_RAD

    def _calculate_fitness(self, lander, dt, crash_reason=None, ignore_crash_penalty=False):
        """
        Calculates the fitness score for a lander based on its performance.
        Higher is better.
        Args:
            lander (Lander): The lander object.
            dt (float): Delta time, the time elapsed since the last frame.
            crash_reason (str, optional): 'ground_crash', 'angle_crash', 'out_of_bounds_crash', 'early_top_crash'.
                                        None if not crashed or soft landed.
            ignore_crash_penalty (bool): If True, crash penalties are not applied.
        Returns:
            tuple: (fitness_score, is_successful_landing_criteria_met)
        """
        # Constants for fitness calculation
        PROXIMITY_H_WEIGHT = 5.0
        PROXIMITY_V_WEIGHT = 3.0
        SUCCESSFUL_LANDING_BONUS = 1000.0
        SOFT_LANDING_OUTSIDE_LZ_BONUS = 100.0
        TIME_SURVIVAL_BONUS_MAX = 50.0 # Max bonus for surviving full time

        # Calculate proximity to landing zone (horizontal and vertical)
        landing_zone_center_x = self.terrain.landing_zone_x + self.terrain.landing_zone_width / 2
        horizontal_dist = abs(lander.x - landing_zone_center_x)
        vertical_dist = abs(lander.y - self.terrain.landing_zone_y)

        # Normalize distances (closer is higher score). Max possible values for normalization.
        max_h_dist = self.screen_width / 2 # From screen center to edge
        max_v_dist = self.screen_height # From top to bottom

        proximity_h_score = 1.0 - (horizontal_dist / max_h_dist)
        proximity_h_score = max(0.0, proximity_h_score) # Clamp to 0-1
        proximity_v_score = 1.0 - (vertical_dist / max_v_dist)
        proximity_v_score = max(0.0, proximity_v_score) # Clamp to 0-1

        # Base fitness for any agent based on proximity.
        fitness = (proximity_h_score * PROXIMITY_H_WEIGHT) + (proximity_v_score * PROXIMITY_V_WEIGHT)

        # Add bonus for time survived (proportional to max simulation time)
        # This rewards agents that fly longer, even if they eventually crash.
        time_survival_ratio = lander.time_elapsed_ms / MAX_SIMULATION_TIME_MS
        fitness += time_survival_ratio * TIME_SURVIVAL_BONUS_MAX

        is_successful_landing_criteria_met = False
        is_soft_landing_anywhere = self._is_soft_landing(lander) # Check for soft landing criteria

        # Add cumulative proximity reward (NEW)
        fitness += lander.cumulative_proximity_score * PROXIMITY_OVER_TIME_REWARD_COEFF

        # Scenario 1: Successfully landed IN the landing zone
        if lander.landed and self.terrain.is_in_landing_zone(lander) and is_soft_landing_anywhere:
            is_successful_landing_criteria_met = True
            fitness += SUCCESSFUL_LANDING_BONUS # Large bonus for perfect landing

            # Reward for low horizontal velocity
            norm_vx = abs(lander.vx) / MAX_LANDING_VX
            fitness += (1.0 - min(1.0, norm_vx)) * 100.0 # Higher points for zero vx

            # Reward for low vertical velocity (soft landing)
            norm_vy = abs(lander.vy) / MAX_LANDING_VY
            fitness += (1.0 - min(1.0, norm_vy)) * 100.0 # Higher points for zero vy

            # Reward for level landing (angle close to 0)
            norm_angle = abs(lander.angle) / MAX_LANDING_ANGLE_RAD
            fitness += (1.0 - min(1.0, norm_angle)) * 100.0 # Higher points for perfect angle

            # Fuel efficiency bonus ONLY for successful landings in LZ
            fuel_used = lander.initial_fuel - lander.fuel
            normalized_fuel_used = fuel_used / lander.initial_fuel
            fitness += (1.0 - normalized_fuel_used) * 50.0 # Max 50 points for full fuel

            return fitness, True

        # Scenario 2: Crashed (anywhere) - now with differentiated penalties
        if lander.crashed:
            if ignore_crash_penalty:
                # If ignoring penalty, just return the base fitness without crash penalties
                # This means it gets the proximity and time survival bonuses, but no negative crash score.
                # We should ensure it doesn't get the SUCCESSFUL_LANDING_BONUS if it actually crashed.
                # So, it's just the 'flying' fitness.
                # print("Ignoring crash penalty for agent.") # Removed for less console spam
                return fitness, False # Not a successful landing, but no penalty
            
            # Original crash penalty logic if not ignoring
            if crash_reason == 'ground_crash':
                impact_speed = math.sqrt(lander.vx**2 + lander.vy**2)
                
                # Normalize impact speed against a threshold where we consider it a very hard crash
                normalized_impact_for_penalty = min(1.0, impact_speed / CRITICAL_IMPACT_SPEED)
                
                # The penalty starts at GROUND_CRASH_BASE_PENALTY and increases up to (BASE + MAX_ADDITIONAL)
                # Note: penalties are negative, so we subtract to make them more negative.
                velocity_scaled_penalty = normalized_impact_for_penalty * GROUND_CRASH_VELOCITY_PENALTY_MAX
                fitness += GROUND_CRASH_BASE_PENALTY - velocity_scaled_penalty
                
            elif crash_reason == 'angle_crash':
                fitness += ANGLE_CRASH_PENALTY
            elif crash_reason == 'out_of_bounds_crash':
                fitness += OUT_OF_BOUNDS_CRASH_PENALTY
            elif crash_reason == 'early_top_crash': # Specific penalty for early top crash
                fitness += MIN_FITNESS_CAP_ON_CRASH # Very harsh penalty
            else: # Fallback for unknown crash type (shouldn't happen if logic is correct)
                # Use a generic crash penalty if type is undefined
                fitness += MIN_FITNESS_CAP_ON_CRASH # A very harsh default if type is unknown
            
            # Ensure fitness doesn't go below a certain cap for any crash type
            fitness = max(fitness, MIN_FITNESS_CAP_ON_CRASH)
            return fitness, False

        # Scenario 3: Landed soft and level but OUTSIDE the landing zone
        if lander.landed and not self.terrain.is_in_landing_zone(lander) and is_soft_landing_anywhere:
            fitness += SOFT_LANDING_OUTSIDE_LZ_BONUS # Significant bonus
            # Add smaller bonuses for velocity/angle control, but no fuel bonus
            norm_vx = abs(lander.vx) / MAX_LANDING_VX
            fitness += (1.0 - min(1.0, norm_vx)) * 10.0 # Max 10 points
            norm_vy = abs(lander.vy) / MAX_LANDING_VY
            fitness += (1.0 - min(1.0, norm_vy)) * 10.0 # Max 10 points
            norm_angle = abs(lander.angle) / MAX_LANDING_ANGLE_RAD
            fitness += (1.0 - min(1.0, norm_angle)) * 10.0 # Max 10 points
            return fitness, False # Not a "successful landing" for the purpose of terrain change

        # Scenario 4: Still flying or timed out without crashing
        # Fitness already includes proximity and time_survival_bonus.
        # Add small rewards for controlling velocity/angle while flying.
        norm_vx_flight = abs(lander.vx) / MAX_FLIGHT_VX
        fitness += (1.0 - min(1.0, norm_vx_flight)) * VELOCITY_CONTROL_REWARD_COEFF

        norm_vy_flight = abs(lander.vy) / MAX_FLIGHT_VY
        fitness += (1.0 - min(1.0, norm_vy_flight)) * VELOCITY_CONTROL_REWARD_COEFF

        norm_angular_velocity_flight = abs(lander.angular_velocity) / MAX_FLIGHT_ANGULAR_VELOCITY_RAD
        fitness += (1.0 - min(1.0, norm_angular_velocity_flight)) * ANGULAR_VELOCITY_CONTROL_REWARD_COEFF

        # Penalize fuel waste for ineffective thrust (NEW)
        # If main thrust is on and lander is still falling quickly (vy > 0.05, i.e., moving down)
        if lander.thrust_main_power_input > 0.1 and lander.vy > 0.05:
            fitness -= (lander.thrust_main_power_input * lander.vy) * FUEL_WASTE_PENALTY_COEFF * dt # Penalize more for higher thrust and faster fall

        return fitness, False # Return current fitness if still flying (partial score for progress)

    def draw_neural_network_overlay(self, screen, genome):
        """
        Draws an overlay visualizing the neural network structure.
        """
        overlay_width = 300
        overlay_height = 200
        overlay_x = self.screen_width - overlay_width - 10
        overlay_y = 50 # Below general info

        # Create a semi-transparent surface for the overlay background
        overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
        overlay_surface.fill((50, 50, 50, 180)) # R, G, B, Alpha (0-255) - 180 is about 70% opaque
        screen.blit(overlay_surface, (overlay_x, overlay_y))

        # Draw overlay border on top of the transparent background
        pygame.draw.rect(screen, LIGHT_GRAY, (overlay_x, overlay_y, overlay_width, overlay_height), 2, 5)

        # Calculate node positions
        node_positions = {}
        input_nodes = sorted([n for n in genome.nodes.values() if n.node_type == 'INPUT'], key=lambda n: n.node_id)
        output_nodes = sorted([n for n in genome.nodes.values() if n.node_type == 'OUTPUT'], key=lambda n: n.node_id)
        hidden_nodes = sorted([n for n in genome.nodes.values() if n.node_type == 'HIDDEN'], key=lambda n: n.node_id)

        node_radius = 6
        padding_x = 20
        padding_y = 20

        # Input nodes
        input_y_step = (overlay_height - 2 * padding_y) / (len(input_nodes) - 1) if len(input_nodes) > 1 else 0
        for i, node in enumerate(input_nodes):
            node_positions[node.node_id] = (overlay_x + padding_x, overlay_y + padding_y + i * input_y_step)

        # Output nodes
        output_y_step = (overlay_height - 2 * padding_y) / (len(output_nodes) - 1) if len(output_nodes) > 1 else 0
        for i, node in enumerate(output_nodes):
            node_positions[node.node_id] = (overlay_x + overlay_width - padding_x, overlay_y + padding_y + i * output_y_step)

        # Hidden nodes (simple horizontal distribution)
        # Distribute hidden nodes across the middle horizontal space
        if hidden_nodes:
            hidden_x_start = overlay_x + padding_x + (overlay_width - 2 * padding_x) / 3
            hidden_x_end = overlay_x + overlay_width - padding_x - (overlay_width - 2 * padding_x) / 3
            hidden_x_step = (hidden_x_end - hidden_x_start) / (len(hidden_nodes) - 1) if len(hidden_nodes) > 1 else 0

            for i, node in enumerate(hidden_nodes):
                # Simple vertical centering for hidden nodes
                node_positions[node.node_id] = (hidden_x_start + i * hidden_x_step, overlay_y + overlay_height / 2)


        # Draw connections
        for conn in genome.connections.values():
            if conn.enabled:
                from_pos = node_positions.get(conn.from_node_id)
                to_pos = node_positions.get(conn.to_node_id)
                if from_pos and to_pos:
                    # Color based on weight: green for positive, red for negative
                    line_color = GREEN if conn.weight > 0 else RED
                    # Thickness based on absolute weight
                    line_thickness = max(1, int(abs(conn.weight) * 2))
                    pygame.draw.line(screen, line_color, from_pos, to_pos, line_thickness)

        # Draw nodes
        for node_id, pos in node_positions.items():
            node = genome.nodes[node_id]
            # Color based on activation value (e.g., blue for low, yellow for high)
            # Normalize activation value (sigmoid outputs 0-1, tanh -1 to 1)
            norm_val = (node.value + 1) / 2 if node.activation_function == 'tanh' else node.value
            node_color = (int(255 * (1 - norm_val)), int(255 * norm_val), 0) # Interpolate between blue and yellow/green
            if node.node_type == 'INPUT': node_color = BLUE
            elif node.node_type == 'OUTPUT': node_color = (255, 165, 0) # Orange for output
            elif node.node_type == 'HIDDEN': node_color = (150, 0, 150) # Purple for hidden nodes with Leaky ReLU

            pygame.draw.circle(screen, node_color, (int(pos[0]), int(pos[1])), node_radius)
            pygame.draw.circle(screen, BLACK, (int(pos[0]), int(pos[1])), node_radius, 1) # Border

    def draw_population_stats_overlay(self, screen, population_stats_data):
        """
        Draws an overlay showing the fitness statistics of each population in a table format.
        population_stats_data is a list of dictionaries, each containing:
        {'pop_idx': int, 'current_best': float, 'avg_gen': float, 'all_time_best': float}
        """
        # Define table dimensions and padding
        cell_height = 25
        header_height = 30
        padding_x = 10
        padding_y = 10
        
        # Column headers and their approximate widths
        headers = ["Pop", "Current Best", "Avg Gen", "All-Time Best"]
        col_widths = [
            self.font.size(headers[0])[0] + 20, # Pop
            self.font.size(headers[1])[0] + 20, # Current Best
            self.font.size(headers[2])[0] + 20, # Avg Gen
            self.font.size(headers[3])[0] + 20  # All-Time Best
        ]

        # Ensure column widths are large enough for data
        for stats in population_stats_data:
            col_widths[0] = max(col_widths[0], self.font.size(f"{stats['pop_idx'] + 1} (Rank {stats['rank'] + 1})")[0] + 20)
            col_widths[1] = max(col_widths[1], self.font.size(f"{stats['current_best']:.2f}")[0] + 20)
            col_widths[2] = max(col_widths[2], self.font.size(f"{stats['avg_gen']:.2f}")[0] + 20)
            col_widths[3] = max(col_widths[3], self.font.size(f"{stats['all_time_best']:.2f}")[0] + 20)

        table_width = sum(col_widths)
        table_height = header_height + len(population_stats_data) * cell_height

        overlay_width = table_width + 2 * padding_x
        overlay_height = table_height + 2 * padding_y
        overlay_x = 10
        overlay_y = 50

        # Create a semi-transparent surface for the overlay background
        overlay_surface = pygame.Surface((overlay_width, overlay_height), pygame.SRCALPHA)
        overlay_surface.fill((50, 50, 50, 180)) # R, G, B, Alpha (0-255) - 180 is about 70% opaque
        screen.blit(overlay_surface, (overlay_x, overlay_y))

        # Draw overlay border on top of the transparent background
        pygame.draw.rect(screen, LIGHT_GRAY, (overlay_x, overlay_y, overlay_width, overlay_height), 2, 5)

        # Draw table headers
        current_x = overlay_x + padding_x
        header_y = overlay_y + padding_y
        for i, header_text in enumerate(headers):
            text_surface = self.font.render(header_text, True, WHITE)
            text_rect = text_surface.get_rect(center=(current_x + col_widths[i] / 2, header_y + header_height / 2))
            screen.blit(text_surface, text_rect)
            current_x += col_widths[i]
        
        # Draw horizontal line after header
        pygame.draw.line(screen, LIGHT_GRAY, 
                         (overlay_x + padding_x, header_y + header_height), 
                         (overlay_x + padding_x + table_width, header_y + header_height), 1)

        # Draw table rows
        current_y = header_y + header_height
        for rank, stats in enumerate(population_stats_data):
            pop_idx = stats['pop_idx']
            pop_color = self.population_colors[pop_idx]

            current_x = overlay_x + padding_x
            
            # Draw vertical lines for columns
            for i in range(len(col_widths) + 1):
                line_x = overlay_x + padding_x + sum(col_widths[:i])
                pygame.draw.line(screen, DARK_GRAY, (line_x, current_y), (line_x, current_y + cell_height), 1)

            # Draw data for each column
            # Column 1: Pop (Rank)
            pop_text = f"{pop_idx + 1} (R{rank + 1})"
            text_surface = self.font.render(pop_text, True, pop_color)
            text_rect = text_surface.get_rect(center=(current_x + col_widths[0] / 2, current_y + cell_height / 2))
            screen.blit(text_surface, text_rect)
            current_x += col_widths[0]

            # Column 2: Current Best
            text_surface = self.font.render(f"{stats['current_best']:.2f}", True, WHITE)
            text_rect = text_surface.get_rect(center=(current_x + col_widths[1] / 2, current_y + cell_height / 2))
            screen.blit(text_surface, text_rect)
            current_x += col_widths[1]

            # Column 3: Avg Gen
            text_surface = self.font.render(f"{stats['avg_gen']:.2f}", True, WHITE)
            text_rect = text_surface.get_rect(center=(current_x + col_widths[2] / 2, current_y + cell_height / 2))
            screen.blit(text_surface, text_rect)
            current_x += col_widths[2]

            # Column 4: All-Time Best
            text_surface = self.font.render(f"{stats['all_time_best']:.2f}", True, WHITE)
            text_rect = text_surface.get_rect(center=(current_x + col_widths[3] / 2, current_y + cell_height / 2))
            screen.blit(text_surface, text_rect)
            current_x += col_widths[3]

            # Draw horizontal line for the row
            pygame.draw.line(screen, DARK_GRAY, 
                             (overlay_x + padding_x, current_y + cell_height), 
                             (overlay_x + padding_x + table_width, current_y + cell_height), 1)
            current_y += cell_height

        # Draw final vertical lines (rightmost)
        final_x_pos = overlay_x + padding_x + table_width
        pygame.draw.line(screen, DARK_GRAY, (final_x_pos, header_y), (final_x_pos, current_y), 1)


    def evaluate_single_genome_fitness(self, genome_to_evaluate):
        """
        Runs a headless simulation for a single genome to determine its fitness.
        This is used for evaluating mutations in a "gradient descent" manner.
        """
        lander = Lander(self.lander_start_x, self.lander_start_y)
        lander.reset(self.lander_start_x, self.lander_start_y) # Reset lander state for this evaluation

        current_time_ms = 0
        dt_eval = 16 # Fixed small dt for consistent evaluation (approx 60 FPS)

        while not lander.landed and not lander.crashed and current_time_ms < MAX_SIMULATION_TIME_MS:
            nn_inputs = self._get_lander_state_for_nn(lander)
            nn_outputs = genome_to_evaluate.forward(nn_inputs, normalization_method='none')

            lander.thrust_main_power_input = max(0.0, min(1.0, nn_outputs[0]))
            lander.angular_velocity_target_input = max(-1.0, min(1.0, nn_outputs[1]))

            lander.update(dt_eval)
            current_time_ms += dt_eval

            # Check for crashes during this mini-simulation
            crash_occurred = False
            crash_type = None
            if lander.y < 0:
                crash_occurred = True
                crash_type = 'early_top_crash' if lander.time_elapsed_ms < 3000 else 'out_of_bounds_crash'
            elif self.terrain.check_collision(lander):
                crash_occurred = True
                crash_type = 'ground_crash'
            elif abs(lander.angle) > MAX_FLIGHT_ANGLE_RAD:
                crash_occurred = True
                crash_type = 'angle_crash'
            elif lander.x < 0 or lander.x > self.screen_width or lander.y > self.screen_height + 50:
                crash_occurred = True
                crash_type = 'out_of_bounds_crash'
            
            if crash_occurred:
                lander.crashed = True
                lander.crash_reason = crash_type
                break # End mini-simulation on crash

        # Calculate final fitness for this single genome
        # No 'ignore_crash_penalty' here, as this is the direct fitness evaluation for mutation acceptance
        fitness, _ = self._calculate_fitness(lander, dt_eval, crash_reason=lander.crash_reason)
        return fitness


    def run_generation_simulation(self): # Removed genomes parameter
        """
        Runs the simulation for all agents across all populations simultaneously.
        Assigns fitness scores back to each genome.
        """
        self.successful_landings_count_current_gen = 0

        all_agents_in_gen = []
        for pop_idx, population in enumerate(self.neat_ga.populations):
            for genome in population:
                lander = Lander(self.lander_start_x, self.lander_start_y)
                lander.reset(self.lander_start_x, self.lander_start_y)
                # Assign population color to the lander
                lander.color = self.population_colors[pop_idx]
                all_agents_in_gen.append({'genome': genome, 'lander': lander, 'finished': False, 'fitness': 0.0, 'population_idx': pop_idx})

        all_finished = False
        # Initialize current_best_genome_in_gen and max_fitness_in_gen for tracking within this simulation run
        current_best_genome_in_gen = None
        max_fitness_in_gen = -float('inf')

        # New: Track current best fitness for each population within this simulation run
        current_best_fitness_per_population = {i: -float('inf') for i in range(self.neat_ga.num_populations)}


        while not all_finished:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    # Assign current fitnesses before exiting
                    for agent_data in all_agents_in_gen:
                        agent_data['genome'].fitness = agent_data['fitness']
                    sys.exit()
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        self.show_nn_overlay = not self.show_nn_overlay
                    if event.key == pygame.K_ESCAPE:
                        print("Escape pressed. Saving best network and exiting.")
                        if self.best_genome_for_display:
                            save_ga_state(self.neat_ga, "neat_ga_state.pkl") # Save GA state on exit
                        pygame.quit()
                        sys.exit()

            num_finished = 0
            # Reset for real-time average fitness calculation
            population_fitness_sums = {i: 0.0 for i in range(self.neat_ga.num_populations)}
            population_agent_counts = {i: 0 for i in range(self.neat_ga.num_populations)}

            # Count currently finished agents
            for agent_data in all_agents_in_gen:
                if agent_data['finished']:
                    num_finished += 1
                    # Include finished agents in real-time average calculation
                    population_fitness_sums[agent_data['population_idx']] += agent_data['fitness']
                    population_agent_counts[agent_data['population_idx']] += 1
            
            num_active_this_iteration = len(all_agents_in_gen) - num_finished

            for agent_data in all_agents_in_gen:
                if agent_data['finished']:
                    continue

                genome = agent_data['genome']
                lander = agent_data['lander']

                # Update cumulative proximity score for this frame
                landing_zone_center_x = self.terrain.landing_zone_x + self.terrain.landing_zone_width / 2
                horizontal_dist = abs(lander.x - landing_zone_center_x)
                vertical_dist = abs(lander.y - self.terrain.landing_zone_y)
                # Normalize distances for proximity reward
                max_h_dist = self.screen_width / 2
                max_v_dist = self.screen_height
                proximity_h_score = 1.0 - (horizontal_dist / max_h_dist)
                proximity_h_score = max(0.0, proximity_h_score)
                proximity_v_score = 1.0 - (vertical_dist / max_v_dist)
                proximity_v_score = max(0.0, proximity_v_score)
                lander.cumulative_proximity_score += (proximity_h_score + proximity_v_score) * dt


                nn_inputs = self._get_lander_state_for_nn(lander)

                # Get NN output
                nn_outputs = genome.forward(nn_inputs, normalization_method='none')

                # Apply NN output to lander controls (continuous outputs)
                # Clamp main thrust output to [0, 1] (sigmoid output)
                lander.thrust_main_power_input = max(0.0, min(1.0, nn_outputs[0]))
                # Clamp angular velocity target input to [-1, 1] (tanh output)
                lander.angular_velocity_target_input = max(-1.0, min(1.0, nn_outputs[1]))

                # Update lander physics
                lander.update(dt)

                crash_occurred = False
                crash_type = None
                
                # Check for collision or timeout
                if lander.y < 0: # Crashed into top of screen
                    if lander.time_elapsed_ms < 3000: # 3 seconds = 3000 ms
                        crash_occurred = True
                        crash_type = 'early_top_crash'
                    else:
                        crash_occurred = True
                        crash_type = 'out_of_bounds_crash'
                elif self.terrain.check_collision(lander):
                    crash_occurred = True
                    crash_type = 'ground_crash'
                # NEW CONDITION: Crash if angle exceeds MAX_FLIGHT_ANGLE_RAD (90 degrees)
                elif abs(lander.angle) > MAX_FLIGHT_ANGLE_RAD:
                    crash_occurred = True
                    crash_type = 'angle_crash'
                elif lander.x < 0 or lander.x > self.screen_width or lander.y > self.screen_height + 50: # Out of bounds (excluding top, handled above)
                    crash_occurred = True
                    crash_type = 'out_of_bounds_crash'
                elif lander.time_elapsed_ms >= MAX_SIMULATION_TIME_MS:
                    # Timeout is not a "crash" in the penalty sense, so no crash_type
                    # But it does mark the agent as finished.
                    agent_data['finished'] = True
                    fitness, _ = self._calculate_fitness(lander, dt, crash_reason=None) # Calculate partial fitness without crash penalty
                    agent_data['fitness'] = fitness
                    # Update current best for this population
                    current_best_fitness_per_population[agent_data['population_idx']] = max(current_best_fitness_per_population[agent_data['population_idx']], agent_data['fitness'])
                    
                    # Update overall best genome for display
                    if fitness > max_fitness_in_gen:
                        max_fitness_in_gen = fitness
                        current_best_genome_in_gen = genome
                    continue # Move to next agent

                if crash_occurred:
                    lander.crashed = True
                    lander.color = RED
                    lander.crash_reason = crash_type # Store the specific crash reason

                    # Check if this is among the last 10 active agents before it crashes
                    if num_active_this_iteration <= 10 and crash_type != 'early_top_crash': # Don't remove penalty for early top crash, it's explicitly penalized
                        # If this is among the last 10 agents and it crashed, remove the crash penalty.
                        fitness, landed_successfully = self._calculate_fitness(lander, dt, crash_reason=None, ignore_crash_penalty=True) # Pass None for crash_type and True for ignore_crash_penalty
                        print(f"Agent in last 10 crashed. Penalty removed. Original crash type: {crash_type}")
                    else:
                        fitness, landed_successfully = self._calculate_fitness(lander, dt, crash_reason=crash_type)

                    if landed_successfully:
                        lander.landed = True
                        lander.color = GREEN
                        self.successful_landings_count_current_gen += 1
                    
                    agent_data['finished'] = True
                    agent_data['fitness'] = fitness
                    # Update current best for this population
                    current_best_fitness_per_population[agent_data['population_idx']] = max(current_best_fitness_per_population[agent_data['population_idx']], agent_data['fitness'])
                    
                    # Update overall best genome for display
                    if fitness > max_fitness_in_gen:
                        max_fitness_in_gen = fitness
                        current_best_genome_in_gen = genome

            # Recalculate num_finished after processing all agents in this iteration
            num_finished = sum(1 for agent_data in all_agents_in_gen if agent_data['finished'])

            # If no agents are finished yet, but we need a best_genome_in_gen for display,
            # pick the one with the highest current (unfinalized) fitness.
            if current_best_genome_in_gen is None and all_agents_in_gen:
                # Find the agent with the highest 'current' fitness (even if not finalized)
                # This ensures there's always a 'best' to display if agents are still flying.
                temp_best_agent = max(all_agents_in_gen, key=lambda ad: ad['fitness'])
                current_best_genome_in_gen = temp_best_agent['genome']
                max_fitness_in_gen = temp_best_agent['fitness']


            # Prepare data for population stats overlay
            population_stats_for_display = []
            for pop_idx in range(self.neat_ga.num_populations):
                avg_f_this_gen = 0.0
                if population_agent_counts[pop_idx] > 0:
                    avg_f_this_gen = population_fitness_sums[pop_idx] / population_agent_counts[pop_idx]
                
                population_stats_for_display.append({
                    'pop_idx': pop_idx,
                    'current_best': current_best_fitness_per_population[pop_idx],
                    'avg_gen': avg_f_this_gen,
                    'all_time_best': self.neat_ga.all_time_best_fitness_per_population[pop_idx]
                })
            
            # Sort for display by current best fitness and add rank
            population_stats_for_display.sort(key=lambda x: x['current_best'], reverse=True)
            for i, stats in enumerate(population_stats_for_display):
                stats['rank'] = i


            if num_finished == len(all_agents_in_gen):
                all_finished = True

            # Rendering (only if enabled)
            if self.render_enabled:
                self.screen.fill(BLACK)
                self.terrain.draw(self.screen)

                # Draw all active landers, setting their color based on state
                for agent_data in all_agents_in_gen:
                    lander_to_draw = agent_data['lander']
                    # Determine color based on final state or "best" status
                    if lander_to_draw.landed and self.terrain.is_in_landing_zone(lander_to_draw) and self._is_soft_landing(lander_to_draw):
                        lander_to_draw.color = GREEN # Successful landing in LZ
                    elif lander_to_draw.landed and not self.terrain.is_in_landing_zone(lander_to_draw) and self._is_soft_landing(lander_to_draw):
                        lander_to_draw.color = YELLOW # Soft landing outside LZ
                    elif lander_to_draw.crashed:
                        # Color already set to RED when crashed, no need to change here
                        pass
                    elif agent_data['genome'] is current_best_genome_in_gen: # Check if it's the current best flying agent
                        lander_to_draw.color = BLUE
                    else: # Default color for other flying agents (their population color)
                        lander_to_draw.color = self.population_colors[agent_data['population_idx']]
                    lander_to_draw.draw(self.screen)

                # Display info
                info_text = self.font.render(
                    f"Global Gen: {self.neat_ga.global_generation} Active Agents: {len(all_agents_in_gen) - num_finished}/{len(all_agents_in_gen)}",
                    True, WHITE
                )
                self.screen.blit(info_text, (10, 10))

                # Display status of the best agent (if available)
                if current_best_genome_in_gen: # Use current_best_genome_in_gen for status
                    best_lander_status = "N/A" # Initialize here, will be updated below
                    # Find the lander associated with the current_best_genome_in_gen
                    # Now, since current_best_genome_in_gen is a direct reference, this comparison works
                    for agent_data in all_agents_in_gen:
                        if agent_data['genome'] is current_best_genome_in_gen: # Use 'is' for identity comparison
                            if agent_data['lander'].landed and self.terrain.is_in_landing_zone(agent_data['lander']) and self._is_soft_landing(agent_data['lander']):
                                best_lander_status = "BEST LANDED (LZ)!"
                            elif agent_data['lander'].landed and not self.terrain.is_in_landing_zone(agent_data['lander']) and self._is_soft_landing(agent_data['lander']):
                                best_lander_status = "BEST LANDED (OUTSIDE LZ)!"
                            elif agent_data['lander'].crashed:
                                # Determine specific crash message for the best agent
                                if agent_data['lander'].crash_reason == 'early_top_crash':
                                    best_lander_status = "BEST CRASHED (EARLY TOP)!"
                                elif agent_data['lander'].crash_reason == 'out_of_bounds_crash':
                                    best_lander_status = "BEST CRASHED (OUT OF BOUNDS)!"
                                elif agent_data['lander'].crash_reason == 'angle_crash':
                                    best_lander_status = "BEST CRASHED (ANGLE)!"
                                else:
                                    best_lander_status = "BEST CRASHED (GROUND)!"
                            elif agent_data['finished']:
                                best_lander_status = "BEST TIME OUT!"
                            else:
                                best_lander_status = "BEST FLYING"
                            break
                    status_render = self.font.render(f"Best: {best_lander_status}", True, YELLOW)
                    self.screen.blit(status_render, (self.screen_width - status_render.get_width() - 10, 10))


                # Draw the best neural network overlay ONLY if toggled on
                if self.show_nn_overlay and current_best_genome_in_gen: # Use current_best_genome_in_gen for drawing
                    self.draw_neural_network_overlay(self.screen, current_best_genome_in_gen)

                # Draw the population stats overlay
                self.draw_population_stats_overlay(self.screen, population_stats_for_display)

                pygame.display.flip()

        # After all agents are finished, store the successful landings count for the next generation's check
        self.previous_successful_landings_count = self.successful_landings_count_current_gen
        
        # --- Handle immediate replacement for early top crashes ---
        for agent_data in all_agents_in_gen:
            if agent_data['lander'].crash_reason == 'early_top_crash':
                pop_idx = agent_data['population_idx']
                original_genome = agent_data['genome'] # This is the reference to the genome object

                # Find the index of this original_genome in its population list
                try:
                    genome_idx_in_pop = self.neat_ga.populations[pop_idx].index(original_genome)
                    # Replace it with a new random genome
                    new_random_genome = self.neat_ga.get_new_random_genome()
                    self.neat_ga.populations[pop_idx][genome_idx_in_pop] = new_random_genome
                    # Reset its fitness to 0 for the next generation's evaluation
                    # This ensures it doesn't negatively impact the *next* generation's selection
                    self.neat_ga.populations[pop_idx][genome_idx_in_pop].fitness = 0.0
                    print(f"Agent in Pop {pop_idx+1} (index {genome_idx_in_pop}) replaced due to early top crash.")
                except ValueError:
                    # This should ideally not happen if the agent_data['genome'] is a direct reference
                    # to an object within self.neat_ga.populations.
                    print(f"Error: Could not find original genome in population {pop_idx} for replacement.")
        # --- End immediate replacement ---

        # Assign fitnesses back to genomes in their respective populations.
        # This is already done within the loop for active_agents.
        # The return value is the overall best genome for the simulation to track.
        overall_best_genome_this_gen = None
        overall_max_fitness_this_gen = -float('inf')
        for population in self.neat_ga.populations:
            if population:
                best_in_pop = max(population, key=lambda g: g.fitness)
                if best_in_pop.fitness > overall_max_fitness_this_gen:
                    overall_max_fitness_this_gen = best_in_pop.fitness
                    overall_best_genome_this_gen = best_in_pop.clone()
        return overall_best_genome_this_gen # Return the overall best genome for tracking


    def run_neat_evolution(self): # Removed generations parameter, now infinite loop
        """
        Main loop to run the NEAT evolution with the Pygame simulation.
        """
        # Initialize best_genome_for_display with the best from the initial setup
        overall_best_initial = None
        max_initial_fitness = -float('inf')
        for population in self.neat_ga.populations:
            if population:
                best_in_pop = max(population, key=lambda g: g.fitness)
                if best_in_pop.fitness > max_initial_fitness:
                    max_initial_fitness = best_in_pop.fitness
                    overall_best_initial = best_in_pop.clone()
        self.best_genome_for_display = overall_best_initial


        while True: # Run for an unlimited number of global generations
            # Update GA's global generation attribute (already done inside evolve)

            # Conditional terrain generation:
            # Only generate new terrain if it's not the very first global generation
            # AND a majority (e.g., >= 50%) of agents successfully landed in the previous generation.
            # This logic now applies to the *total* successful landings across all populations.
            total_agents = self.neat_ga.num_populations * self.neat_ga.agents_per_population
            if self.neat_ga.global_generation > 0 and self.previous_successful_landings_count >= total_agents * 0.5:
                self.terrain.generate_terrain()
                if self.render_enabled:
                    print(f"--- Global Generation {self.neat_ga.global_generation + 1}: Terrain Randomized! ---")
            else:
                if self.render_enabled:
                    print(f"--- Global Generation {self.neat_ga.global_generation + 1}: Keeping current terrain. ---")

            # Reset per-generation state for terrain randomization
            # self.successful_landings_count_current_gen is reset in run_generation_simulation
            self.terrain_randomized_this_gen = False
            self.current_generation_number = self.neat_ga.global_generation # Update current global generation number

            # Run simulation for all agents across all populations
            # The fitnesses are assigned directly within run_generation_simulation
            overall_best_genome_this_gen_from_sim = self.run_generation_simulation()

            # Evolve all populations, passing the fitness evaluator for mutation acceptance
            overall_best_genome_after_evolve = self.neat_ga.evolve(self.evaluate_single_genome_fitness)

            # Update the best genome for display (overall best found so far)
            if not self.best_genome_for_display or overall_best_genome_after_evolve.fitness > self.best_genome_for_display.fitness:
                self.best_genome_for_display = overall_best_genome_after_evolve.clone()

            # --- Auto-save GA state after every 10 global generations ---
            if self.neat_ga.global_generation % 10 == 0:
                print(f"--- Auto-saving GA state at Global Generation {self.neat_ga.global_generation} ---")
                save_ga_state(self.neat_ga, "neat_ga_state.pkl")
            # --- End Auto-save ---

            # No explicit 'gen += 1' here, as it's handled by self.neat_ga.global_generation increment in evolve.

        # Save the best genome on exit (this part will only be reached if sys.exit() is called)
        if self.best_genome_for_display:
            save_ga_state(self.neat_ga, "neat_ga_state.pkl") # Save GA state on exit
            print("Best neural network saved to best_lander_nn.txt")

        pygame.quit()

# --- Save/Load Functions ---

def save_ga_state(ga_instance, filename):
    """Saves the entire GeneticAlgorithm instance state and global innovation tracker to a file using pickle."""
    try:
        # Create a dictionary to hold the GA instance and the global innovation tracker
        state_to_save = {
            'ga_instance': ga_instance,
            'innovation_tracker': _innovation_tracker
        }
        with open(filename, 'wb') as f: # Open in binary write mode
            pickle.dump(state_to_save, f)
        print(f"GeneticAlgorithm state saved to {filename}")
        return True
    except IOError as e:
        print(f"Error saving GA state to {filename}: {e}")
        return False
    except Exception as e:
        print(f"An unexpected error occurred during GA state saving: {e}")
        return False

def load_ga_state(filename):
    """Loads the entire GeneticAlgorithm instance state and global innovation tracker from a file using pickle."""
    global _innovation_tracker # Declare intent to modify the global variable
    try:
        with open(filename, 'rb') as f: # Open in binary read mode
            loaded_state = pickle.load(f)
        
        # Extract the GA instance and the innovation tracker
        loaded_ga_instance = loaded_state['ga_instance']
        loaded_innovation_tracker = loaded_state['innovation_tracker']

        # Update the global innovation tracker with the loaded one
        _innovation_tracker.clear() # Clear existing global tracker
        _innovation_tracker.update(loaded_innovation_tracker) # Update with loaded state

        print(f"GeneticAlgorithm state loaded from {filename}. Resuming from Global Generation {loaded_ga_instance.global_generation}.")
        return loaded_ga_instance
    except FileNotFoundError:
        print(f"No saved state found at {filename}.")
        return None
    except Exception as e:
        print(f"Error loading GA state from {filename}: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # NEAT configuration
    neat_config = {
        'agents_per_population': 20, # Size of each sub-population
        'num_populations': 5, # Number of distinct populations
        'c1': 1.0,  # Coefficient for excess genes
        'c2': 1.0,  # Coefficient for disjoint genes
        'c3': 0.4,  # Coefficient for weight differences
        'compatibility_threshold': 3.0, # Threshold for speciation
        'add_node_prob': 0.05,
        'add_connection_prob': 0.08,
        'weight_mutate_prob': 0.8,
        'weight_replace_prob': 0.1,
        'enable_disable_prob': 0.01,
        'inter_species_breeding_prob': 0.001,
        'survival_rate': 0.2,
        'stagnation_threshold': 15,
        'min_species_size': 2,
        'elitism_species_count': 1
    }

    # Inputs for NN: x, y, vx, vy, angle, angular_velocity, fuel,
    #                dist_to_lz_x, dist_to_lz_y,
    #                terrain_height_at_lander_x, terrain_slope_at_lander_x,
    #                altitude_above_terrain (NEW)
    NN_INPUT_SIZE = 12 # Updated from 11
    # Outputs for NN: main_thrust_power (0-1), angular_velocity_control_target (-1 to 1)
    NN_OUTPUT_SIZE = 2

    # Attempt to load saved GA state
    neat_ga_instance = load_ga_state("neat_ga_state.pkl") # Changed filename extension to .pkl

    if neat_ga_instance is None:
        print("No saved state found or error loading. Initializing new GeneticAlgorithm.")
        neat_ga_instance = GeneticAlgorithm(
            num_populations=neat_config['num_populations'],
            agents_per_population=neat_config['agents_per_population'],
            input_size=NN_INPUT_SIZE,
            output_size=NN_OUTPUT_SIZE,
            config=neat_config,
            initialize_populations=True
        )
    else:
        # If loaded, the global innovation tracker is already updated by load_ga_state.
        # Ensure NN_INPUT_SIZE and NN_OUTPUT_SIZE are consistent with loaded GA
        neat_ga_instance.input_size = NN_INPUT_SIZE
        neat_ga_instance.output_size = NN_OUTPUT_SIZE
        # Re-initialize Species objects if loading from an older format, or ensure they are correctly loaded
        # For simplicity in this update, we assume the Species class structure is consistent
        # If loading from a very old save, it might be necessary to re-speciate or re-initialize species.
        # For now, we'll rely on pickle to handle the Species object structure.
        # Also, ensure mutation rates are set from config or adaptively after load
        neat_ga_instance.config.update(neat_config) # Ensure loaded GA uses current config values


    # Initialize and run the simulation
    # Set render_enabled to False for faster training, True to visualize
    simulation = LunarLanderSimulation(SCREEN_WIDTH, SCREEN_HEIGHT, neat_ga_instance, render_enabled=True)
    simulation.run_neat_evolution() # No generations argument needed here
