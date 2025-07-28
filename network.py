import random
import math
import pygame
import sys # Import sys for clean exit

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

    def to_string(self):
        """Converts NodeGene to a string for serialization."""
        return f"{self.node_id},{self.node_type},{self.activation_function}"

    @staticmethod
    def from_string(s):
        """Creates NodeGene from a string."""
        parts = s.split(',')
        return NodeGene(int(parts[0]), parts[1], parts[2])

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

    def to_string(self):
        """Converts ConnectionGene to a string for serialization."""
        return f"{self.from_node_id},{self.to_node_id},{self.weight},{int(self.enabled)},{self.innovation_number}"

    @staticmethod
    def from_string(s):
        """Creates ConnectionGene from a string."""
        parts = s.split(',')
        return ConnectionGene(int(parts[0]), int(parts[1]), float(parts[2]), bool(int(parts[3])), int(parts[4]))

    def __repr__(self):
        return (f"Conn(Inn:{self.innovation_number}, From:{self.from_node_id} "
                f"To:{self.to_node_id}, W:{self.weight:.2f}, Enabled:{self.enabled})")

# --- NeuralNetwork Class (Genome) ---

class NeuralNetwork:
    """
    Represents a single neural network genome in the NEAT algorithm.
    It manages its nodes, connections, and handles the forward pass.
    """
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.nodes = {}  # {node_id: NodeGene_object}
        self.connections = {} # {innovation_number: ConnectionGene_object}
        self.node_id_counter = 0 # To assign unique IDs to new nodes

        self._initialize_network()
        self.fitness = 0.0 # Placeholder for fitness value

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
            self.nodes[node_id] = NodeGene(node_id, 'HIDDEN')
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
        self.nodes[new_node_id] = NodeGene(new_node_id, 'HIDDEN')

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
        new_nn = NeuralNetwork(self.input_size, self.output_size)
        new_nn.nodes = {node_id: NodeGene(node.node_id, node.node_type, node.activation_function)
                        for node_id, node in self.nodes.items()}
        new_nn.connections = {conn_id: ConnectionGene(conn.from_node_id, conn.to_node_id,
                                                      conn.weight, conn.enabled, conn.innovation_number)
                              for conn_id, conn in self.connections.items()}
        new_nn.node_id_counter = self.node_id_counter
        new_nn.fitness = self.fitness
        return new_nn

    def to_string(self):
        """Serializes the neural network to a string."""
        node_strings = ";".join([node.to_string() for node in self.nodes.values()])
        conn_strings = ";".join([conn.to_string() for conn in self.connections.values()])
        # Also serialize the global innovation tracker's next_innovation_number
        global _innovation_tracker
        return f"{self.input_size},{self.output_size},{self.node_id_counter},{self.fitness},{_innovation_tracker['next_innovation_number']}|{node_strings}|{conn_strings}"

    @staticmethod
    def from_string(s):
        """Deserializes a neural network from a string."""
        try:
            header_part, node_part, conn_part = s.split('|')
            input_size, output_size, node_id_counter, fitness, next_innovation_num = header_part.split(',')

            nn = NeuralNetwork(int(input_size), int(output_size))
            nn.node_id_counter = int(node_id_counter)
            nn.fitness = float(fitness)
            nn.nodes = {}
            nn.connections = {}

            if node_part:
                for node_str in node_part.split(';'):
                    node = NodeGene.from_string(node_str)
                    nn.nodes[node.node_id] = node
            if conn_part:
                for conn_str in conn_part.split(';'):
                    conn = ConnectionGene.from_string(conn_str)
                    nn.connections[conn.innovation_number] = conn

            # Restore global innovation tracker state
            global _innovation_tracker
            _innovation_tracker['next_innovation_number'] = max(_innovation_tracker['next_innovation_number'], int(next_innovation_num))

            return nn
        except Exception as e:
            print(f"Error deserializing NeuralNetwork: {e}")
            return None

    def __repr__(self):
        return (f"NN(Nodes:{len(self.nodes)}, Conns:{len(self.connections)}, "
                f"Enabled:{sum(1 for c in self.connections.values() if c.enabled)}, Fitness:{self.fitness:.2f})")


# --- GeneticAlgorithm Class ---

class GeneticAlgorithm:
    """
    Implements the NEAT genetic algorithm to evolve neural networks.
    Manages populations, speciation, reproduction, and mutation.
    """
    def __init__(self,
                 population_size,
                 input_size,
                 output_size,
                 config=None):
        self.population_size = population_size
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

        self.population = []
        self.species = [] # List of lists of NeuralNetwork objects (each inner list is a species)
        self.generation = 0

        self._initialize_population()

    def _initialize_population(self):
        """Creates the initial population of simple neural networks."""
        for _ in range(self.population_size):
            self.population.append(NeuralNetwork(self.input_size, self.output_size))
        self._speciate() # Initial speciation

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

    def _speciate(self):
        """
        Divides the current population into species based on genetic compatibility.
        Each species has a representative genome.
        """
        new_species = []
        for genome in self.population:
            found_species = False
            for species_members in self.species:
                # The first member of a species is its representative
                representative = species_members[0]
                if self._distance(genome, representative) < self.config['compatibility_threshold']:
                    species_members.append(genome)
                    found_species = True
                    break
            if not found_species:
                # Create a new species with this genome as the representative
                new_species.append([genome])

        # Filter out empty species and update species data (e.g., stagnation)
        self.species = []
        for old_species_members in new_species:
            if not old_species_members:
                continue
            self.species.append(old_species_members)

        # Sort species by the fitness of their best member (for elitism/survival)
        self.species.sort(key=lambda s: max(g.fitness for g in s), reverse=True)

        # Keep only the top performing species (elitism for species)
        self.species = self.species[:max(self.config['elitism_species_count'], len(self.species))]

    def _crossover(self, parent1, parent2):
        """
        Performs crossover between two parent genomes to create a child genome.
        Assumes parent1 is the fitter parent (or equally fit).
        """
        if parent2.fitness > parent1.fitness:
            parent1, parent2 = parent2, parent1 # Ensure parent1 is the fitter one

        child = NeuralNetwork(self.input_size, self.output_size)
        child.nodes = {}
        child.connections = {}

        # Inherit nodes
        all_node_ids = set(parent1.nodes.keys()).union(set(parent2.nodes.keys()))
        for node_id in all_node_ids:
            if node_id in parent1.nodes and node_id in parent2.nodes:
                # Inherit matching node randomly
                child.nodes[node_id] = random.choice([parent1.nodes[node_id], parent2.nodes[node_id]]).clone()
            elif node_id in parent1.nodes:
                # Inherit disjoint/excess node from fitter parent
                child.nodes[node_id] = parent1.nodes[node_id].clone()
            elif node_id in parent2.nodes:
                # Inherit disjoint/excess node from fitter parent (parent1 is fitter)
                # If parent2 has a node that parent1 doesn't, it's only inherited if it's not excess in parent2
                # and if parent1 is not fitter, but we already ensured parent1 is fitter.
                # So, only inherit if it's a matching gene that parent1 just didn't have.
                # In NEAT, disjoint/excess from the *less* fit parent are only inherited with a certain probability.
                # For simplicity, we'll just inherit from parent1 if it has it, otherwise from parent2 if it has it.
                child.nodes[node_id] = parent2.nodes[node_id].clone() # This is a simplification.

        # Inherit connections
        all_innovation_numbers = set(parent1.connections.keys()).union(set(parent2.connections.keys()))
        for inn in all_innovation_numbers:
            conn1 = parent1.connections.get(inn)
            conn2 = parent2.connections.get(inn)

            if conn1 and conn2:
                # Matching connection: inherit randomly or average weights
                if random.random() < 0.5:
                    child.connections[inn] = conn1.clone()
                else:
                    child.connections[inn] = conn2.clone()
                # If one is disabled and the other enabled, child has 75% chance of being enabled
                if not conn1.enabled or not conn2.enabled:
                    if random.random() < 0.75:
                        child.connections[inn].enabled = True
                    else:
                        child.connections[inn].enabled = False
            elif conn1:
                # Disjoint/Excess from fitter parent (parent1)
                child.connections[inn] = conn1.clone()
            elif conn2:
                # Disjoint/Excess from less fit parent (parent2) - inherited with probability
                if random.random() < 0.5: # Example probability, can be configured
                    child.connections[inn] = conn2.clone()

        # Ensure node_id_counter is updated for the child
        child.node_id_counter = max(node.node_id for node in child.nodes.values()) + 1 if child.nodes else 0
        return child

    def _mutate(self, genome):
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
                conn.enabled = not conn.enabled # Toggle enabled status


    def evolve(self, fitness_function, generations=1):
        """
        Evolves the population for a specified number of generations.
        This is the main reproduction and mutation loop.
        """
        for _ in range(generations):
            self.generation += 1
            self._speciate() # Speciate the current population

            new_population = []

            # Calculate total adjusted fitness for proportional selection
            min_best_species_fitness = 0
            if self.species and any(s for s in self.species):
                min_best_species_fitness = min(max(g.fitness for g in s) for s in self.species if s)

            fitness_offset = 0
            if min_best_species_fitness < 0:
                fitness_offset = abs(min_best_species_fitness) + 1.0

            total_adjusted_fitness = 0
            for species_members in self.species:
                if species_members:
                    adjusted_species_best_fitness = max(g.fitness for g in species_members) + fitness_offset
                    total_adjusted_fitness += adjusted_species_best_fitness

            # Handle case where no species or all adjusted fitnesses are zero
            if total_adjusted_fitness <= 0 and self.species:
                print("Warning: total_adjusted_fitness is non-positive. Distributing offspring equally.")
                species_offspring_counts = {}
                for species_members in self.species:
                    species_offspring_counts[id(species_members)] = self.population_size // len(self.species)
                for i in range(self.population_size % len(self.species)):
                    species_offspring_counts[id(self.species[i])] += 1
            elif not self.species:
                print("No species left. Re-initializing population.")
                self.population = [NeuralNetwork(self.input_size, self.output_size) for _ in range(self.population_size)]
                self._speciate()
                return max(self.population, key=lambda g: g.fitness) # Return best of new population
            else:
                species_offspring_counts = {}
                for species_members in self.species:
                    if not species_members: continue
                    species_best_fitness = max(g.fitness for g in species_members)
                    adjusted_species_best_fitness = species_best_fitness + fitness_offset
                    adjusted_species_best_fitness = max(0.0, adjusted_species_best_fitness) # Ensure non-negative
                    offspring_count = int(round(adjusted_species_best_fitness / total_adjusted_fitness * self.population_size))
                    species_offspring_counts[id(species_members)] = offspring_count

            current_offspring_sum = sum(species_offspring_counts.values())
            if current_offspring_sum < self.population_size:
                remaining = self.population_size - current_offspring_sum
                for _ in range(remaining):
                    if self.species:
                        best_species_id = id(self.species[0])
                        species_offspring_counts[best_species_id] = species_offspring_counts.get(best_species_id, 0) + 1
            elif current_offspring_sum > self.population_size:
                excess = current_offspring_sum - self.population_size
                for _ in range(excess):
                    if self.species:
                        worst_species_id = id(self.species[-1])
                        if species_offspring_counts.get(worst_species_id, 0) > 0:
                            species_offspring_counts[worst_species_id] -= 1

            for species_members in self.species:
                species_members.sort(key=lambda g: g.fitness, reverse=True)
                num_survivors = max(1, int(len(species_members) * self.config['survival_rate']))
                for i in range(min(num_survivors, len(species_members))):
                    new_population.append(species_members[i].clone())

                current_species_id = id(species_members)
                num_offspring_to_create = species_offspring_counts.get(current_species_id, 0) - num_survivors
                if num_offspring_to_create < 0: num_offspring_to_create = 0

                for _ in range(num_offspring_to_create):
                    parent1 = random.choice(species_members[:num_survivors])
                    parent2 = random.choice(species_members[:num_survivors])
                    child = self._crossover(parent1, parent2)
                    self._mutate(child)
                    new_population.append(child)

            while len(new_population) < self.population_size:
                new_population.append(NeuralNetwork(self.input_size, self.output_size))

            self.population = new_population[:self.population_size]

            best_genome = max(self.population, key=lambda g: g.fitness)
            print(f"Generation {self.generation} complete. Best fitness: {best_genome.fitness:.4f}")
            print(f"Number of species: {len(self.species)}")

        return max(self.population, key=lambda g: g.fitness)


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

# Lander physics constants
GRAVITY = 0.0005 # Pixels per ms^2, adjusted for Pygame's typical scale
MAIN_THRUST_POWER = 0.002
SIDE_THRUST_POWER = 0.00005 # Reduced side thrust power
ANGULAR_DRAG = 0.95 # Increased angular drag (closer to 0 for more drag)
LINEAR_DRAG = 0.999 # Multiplier for linear velocity each frame

INITIAL_FUEL = 30000 # ~30 seconds at 1 unit/ms main thrust consumption
FUEL_CONSUMPTION_MAIN = 1.0 # Units per ms
FUEL_CONSUMPTION_SIDE = 0.1 # Units per ms

MAX_SIMULATION_TIME_MS = 45000 # Increased from 30000 (30 seconds to 45 seconds)

# Landing criteria
MAX_LANDING_VX = 0.05
MAX_LANDING_VY = 0.1
MAX_LANDING_ANGLE_RAD = math.radians(5) # +/- 5 degrees for soft landing

# New constant for maximum flight angle before crashing
MAX_FLIGHT_ANGLE_RAD = math.radians(90) # +/- 90 degrees for crash during flight

# New constant for limiting side thruster angular acceleration
# 5 degrees per second squared, converted to radians per millisecond squared
MAX_THRUSTER_ANGULAR_ACCELERATION_RAD_PER_MS2 = math.radians(5) / (1000.0 * 1000.0)


class Lander:
    def __init__(self, start_x, start_y, initial_fuel=INITIAL_FUEL):
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = math.radians(0) # Radians, 0 is straight up
        self.angular_velocity = 0.0
        self.fuel = initial_fuel
        self.initial_fuel = initial_fuel
        self.mass = 1.0 # Base mass
        self.base_mass = 1.0
        self.fuel_mass_ratio = 0.00005 # How much fuel affects mass (e.g., 1 unit of fuel is 0.00005 mass units)

        self.width = 20
        self.height = 30
        self.color = WHITE # Default color

        self.thrust_main_on = False
        self.thrust_left_on = False
        self.thrust_right_on = False

        self.landed = False
        self.crashed = False
        self.score = 0.0
        self.time_elapsed_ms = 0

    def update(self, dt):
        if self.landed or self.crashed:
            return

        self.time_elapsed_ms += dt

        # Update mass based on fuel
        self.mass = self.base_mass + (self.fuel * self.fuel_mass_ratio)

        # Apply thrust
        force_x = 0.0
        force_y = 0.0
        
        # Calculate side thruster torque separately to apply limits
        side_thruster_torque_force = 0.0 # This is the force component for torque
        if self.thrust_left_on and self.fuel > 0:
            side_thruster_torque_force -= SIDE_THRUST_POWER
            self.fuel = max(0, self.fuel - FUEL_CONSUMPTION_SIDE * dt)

        if self.thrust_right_on and self.fuel > 0:
            side_thruster_torque_force += SIDE_THRUST_POWER
            self.fuel = max(0, self.fuel - FUEL_CONSUMPTION_SIDE * dt)

        # Calculate angular acceleration from side thrusters
        # Assuming torque arm is half the lander's height for simplicity
        torque_arm = self.height / 2.0
        angular_accel_from_thrusters = (side_thruster_torque_force * torque_arm) / self.mass

        # Limit the angular acceleration that the thrusters can cause
        # This simulates a thruster with a limited range of movement.
        angular_accel_from_thrusters = max(-MAX_THRUSTER_ANGULAR_ACCELERATION_RAD_PER_MS2,
                                           min(MAX_THRUSTER_ANGULAR_ACCELERATION_RAD_PER_MS2,
                                               angular_accel_from_thrusters))

        # Apply main thrust
        if self.thrust_main_on and self.fuel > 0:
            # Corrected: horizontal force now pushes in the direction the lander is tilted
            force_x += MAIN_THRUST_POWER * math.sin(self.angle) * dt
            force_y -= MAIN_THRUST_POWER * math.cos(self.angle) * dt # Thrust acts upwards relative to lander
            self.fuel = max(0, self.fuel - FUEL_CONSUMPTION_MAIN * dt)

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

        # Update angular velocity and angle
        self.angular_velocity += angular_accel_from_thrusters * dt # Add limited angular acceleration
        self.angular_velocity *= ANGULAR_DRAG # Apply angular drag
        self.angle += self.angular_velocity * dt

        # Keep angle within -pi to pi range
        self.angle = (self.angle + math.pi) % (2 * math.pi) - math.pi

        # Check fuel
        if self.fuel <= 0:
            self.fuel = 0
            self.thrust_main_on = False
            self.thrust_left_on = False
            self.thrust_right_on = False

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
        if self.thrust_main_on and self.fuel > 0:
            flame_length = 10 + random.randint(0, 5) # Flicker effect
            flame_width = self.width / 2

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
            # Corrected: Flame points in the direction of the thrust vector.
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

        # Removed side thruster exhaust graphics as requested
        # if self.thrust_left_on and self.fuel > 0:
        #     # Left thruster (pushes left, so flame goes right relative to lander)
        #     # Position on the lander's right side
        #     thruster_pos_x = self.width * 0.5
        #     thruster_pos_y = side_thruster_offset_y
        #     thruster_origin = rotate_and_translate(thruster_pos_x, thruster_pos_y)
        #     # Flame points outwards (right relative to lander)
        #     flame_tip_x = thruster_origin[0] + side_flame_length * math.cos(self.angle)
        #     flame_tip_y = thruster_origin[1] + side_flame_length * math.sin(self.angle)
        #     pygame.draw.line(screen, ORANGE, thruster_origin, (flame_tip_x, flame_tip_y), side_flame_width)

        # if self.thrust_right_on and self.fuel > 0:
        #     # Right thruster (pushes right, so flame goes left relative to lander)
        #     # Position on the lander's left side
        #     thruster_pos_x = -self.width * 0.5
        #     thruster_pos_y = side_thruster_offset_y
        #     thruster_origin = rotate_and_translate(thruster_pos_x, thruster_pos_y)
        #     # Flame points outwards (left relative to lander)
        #     flame_tip_x = thruster_origin[0] - side_flame_length * math.cos(self.angle)
        #     flame_tip_y = thruster_origin[1] - side_flame_length * math.sin(self.angle)
        #     pygame.draw.line(screen, ORANGE, thruster_origin, (flame_tip_x, flame_tip_y), side_flame_width)


    def reset(self, start_x, start_y):
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.angle = random.uniform(-math.pi/4, math.pi/4) # Small random initial angle
        self.angular_velocity = 0.0
        self.fuel = self.initial_fuel
        self.mass = self.base_mass + (self.fuel * self.fuel_mass_ratio)
        self.thrust_main_on = False
        self.thrust_left_on = False
        self.thrust_right_on = False
        self.landed = False
        self.crashed = False
        self.score = 0.0
        self.time_elapsed_ms = 0
        self.color = WHITE # Reset color on reset

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

    def _get_lander_state_for_nn(self, lander):
        """
        Extracts and normalizes relevant state information for the neural network.
        Inputs: x, y, vx, vy, angle, angular_velocity, fuel,
                distance_to_landing_zone_x, distance_to_lz_y,
                terrain_height_at_lander_x, terrain_slope_at_lander_x
        """
        # Normalize values to be between 0 and 1 or -1 and 1
        norm_x = lander.x / self.screen_width
        norm_y = lander.y / self.screen_height
        norm_vx = lander.vx / (MAX_LANDING_VX * 10) # Max speed for normalization
        norm_vy = lander.vy / (MAX_LANDING_VY * 10)
        norm_angle = lander.angle / math.pi # -1 to 1 range for -pi to pi
        norm_angular_velocity = lander.angular_velocity / math.radians(10) # Max angular velocity for normalization
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

        return [
            norm_x, norm_y,
            norm_vx, norm_vy,
            norm_angle, norm_angular_velocity,
            norm_fuel,
            dist_to_lz_x, dist_to_lz_y,
            norm_terrain_height, norm_terrain_slope
        ]

    def _is_soft_landing(self, lander):
        """Checks if the lander meets the velocity and angle criteria for a soft landing."""
        return abs(lander.vx) < MAX_LANDING_VX and \
               abs(lander.vy) < MAX_LANDING_VY and \
               abs(lander.angle) < MAX_LANDING_ANGLE_RAD

    def _calculate_fitness(self, lander):
        """
        Calculates the fitness score for a lander based on its performance.
        Higher is better.
        Returns a tuple: (fitness_score, is_successful_landing_criteria_met)
        """
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
        # This will be the foundation for all scores.
        fitness = (proximity_h_score * 5.0) + (proximity_v_score * 3.0) # Max 8 points for proximity

        is_successful_landing_criteria_met = False
        is_soft_landing_anywhere = self._is_soft_landing(lander) # Check for soft landing criteria

        # Scenario 1: Successfully landed IN the landing zone
        if lander.landed and self.terrain.is_in_landing_zone(lander) and is_soft_landing_anywhere:
            is_successful_landing_criteria_met = True
            fitness += 1000.0 # Large bonus for perfect landing (make this very high)

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

        # Scenario 2: Crashed (anywhere)
        if lander.crashed:
            # Fitness is primarily proximity-based.
            # Add a significant penalty for crashing to ensure it's lower than soft landings.
            fitness -= 100.0 # Strong penalty for crashing, adjusted to ensure hierarchy
            # Do NOT clamp to 0.0 here, allow negative scores for bad crashes.
            return fitness, False

        # Scenario 3: Landed soft and level but OUTSIDE the landing zone
        if lander.landed and not self.terrain.is_in_landing_zone(lander) and is_soft_landing_anywhere:
            fitness += 100.0 # Significant bonus for soft landing outside LZ, adjusted to ensure hierarchy
            # Add smaller bonuses for velocity/angle control, but no fuel bonus
            norm_vx = abs(lander.vx) / MAX_LANDING_VX
            fitness += (1.0 - min(1.0, norm_vx)) * 10.0 # Max 10 points
            norm_vy = abs(lander.vy) / MAX_LANDING_VY
            fitness += (1.0 - min(1.0, norm_vy)) * 10.0 # Max 10 points
            norm_angle = abs(lander.angle) / MAX_LANDING_ANGLE_RAD
            fitness += (1.0 - min(1.0, norm_angle)) * 10.0 # Max 10 points
            return fitness, False # Not a "successful landing" for the purpose of terrain change

        # Scenario 4: Still flying or timed out without crashing
        # Fitness is base proximity + small rewards for control while flying.
        # These are the lowest non-crash scores, as they didn't even attempt a soft landing.
        norm_vx = abs(lander.vx) / (MAX_LANDING_VX * 5)
        fitness += (1.0 - min(1.0, norm_vx)) * 2.0

        norm_vy = abs(lander.vy) / (MAX_LANDING_VY * 5)
        fitness += (1.0 - min(1.0, norm_vy)) * 2.0

        norm_angle = abs(lander.angle) / math.pi
        fitness += (1.0 - min(1.0, norm_angle)) * 2.0

        return fitness, False # Return current fitness if still flying (partial score for progress)

    def draw_neural_network_overlay(self, screen, genome):
        """
        Draws an overlay visualizing the neural network structure.
        """
        overlay_width = 300
        overlay_height = 200
        overlay_x = self.screen_width - overlay_width - 10
        overlay_y = 50 # Below general info

        # Draw overlay background
        pygame.draw.rect(screen, DARK_GRAY, (overlay_x, overlay_y, overlay_width, overlay_height), 0, 5)
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

            pygame.draw.circle(screen, node_color, (int(pos[0]), int(pos[1])), node_radius)
            pygame.draw.circle(screen, BLACK, (int(pos[0]), int(pos[1])), node_radius, 1) # Border

    def run_generation_simulation(self, genomes):
        """
        Runs the simulation for all agents in a generation simultaneously.
        Returns a list of fitness scores for each genome.
        """
        # Reset successful landings count for the current generation
        self.successful_landings_count_current_gen = 0

        # Create a list of (genome, lander, is_finished) tuples
        active_agents = []
        for genome in genomes:
            lander = Lander(self.lander_start_x, self.lander_start_y)
            lander.reset(self.lander_start_x, self.lander_start_y)
            active_agents.append({'genome': genome, 'lander': lander, 'finished': False, 'fitness': 0.0})

        all_finished = False
        current_best_genome_in_gen = None
        max_fitness_in_gen = -1.0

        while not all_finished:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    # Return current fitnesses if quitting mid-generation
                    return [agent['fitness'] for agent in active_agents]
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_o:
                        self.show_nn_overlay = not self.show_nn_overlay
                    if event.key == pygame.K_ESCAPE: # Handle Escape key
                        print("Escape pressed. Saving best network and exiting.")
                        if self.best_genome_for_display:
                            save_network(self.best_genome_for_display, "best_lander_nn.txt")
                        pygame.quit()
                        sys.exit() # Exit cleanly

            num_finished = 0
            for agent_data in active_agents:
                if agent_data['finished']:
                    num_finished += 1
                    continue

                genome = agent_data['genome']
                lander = agent_data['lander']

                # Get lander state for NN
                nn_inputs = self._get_lander_state_for_nn(lander)

                # Get NN output
                nn_outputs = genome.forward(nn_inputs, normalization_method='none')

                # Apply NN output to lander controls
                main_thrust_output = nn_outputs[0]
                rotation_thrust_output = nn_outputs[1]

                lander.thrust_main_on = main_thrust_output > 0.5
                lander.thrust_left_on = rotation_thrust_output < -0.2
                lander.thrust_right_on = rotation_thrust_output > 0.2

                # Update lander physics
                lander.update(dt)

                # Check for collision or timeout
                if self.terrain.check_collision(lander):
                    fitness, landed_successfully = self._calculate_fitness(lander)
                    if landed_successfully:
                        lander.landed = True
                        lander.color = GREEN
                        self.successful_landings_count_current_gen += 1
                    else:
                        lander.crashed = True
                        lander.color = RED
                    agent_data['finished'] = True
                    agent_data['fitness'] = fitness
                    num_finished += 1
                # NEW CONDITION: Crash if angle exceeds MAX_FLIGHT_ANGLE_RAD (90 degrees)
                elif abs(lander.angle) > MAX_FLIGHT_ANGLE_RAD:
                    lander.crashed = True
                    lander.color = RED
                    agent_data['finished'] = True
                    agent_data['fitness'] = self._calculate_fitness(lander)[0] # Calculate fitness for crash
                    num_finished += 1
                elif lander.x < 0 or lander.x > self.screen_width or lander.y < 0 or lander.y > self.screen_height + 50:
                    lander.crashed = True
                    agent_data['finished'] = True
                    agent_data['fitness'] = self._calculate_fitness(lander)[0] # Calculate fitness for out-of-bounds crash
                    num_finished += 1
                elif lander.time_elapsed_ms >= MAX_SIMULATION_TIME_MS:
                    fitness, _ = self._calculate_fitness(lander) # Calculate partial fitness
                    agent_data['finished'] = True
                    agent_data['fitness'] = fitness
                    num_finished += 1

                # Update current best genome for display during the generation
                # Store the direct genome object reference, not a clone, for correct comparison
                if agent_data['fitness'] > max_fitness_in_gen:
                    max_fitness_in_gen = agent_data['fitness']
                    current_best_genome_in_gen = genome # Store reference to the actual genome object

            if num_finished == len(active_agents):
                all_finished = True

            # Rendering (only if enabled)
            if self.render_enabled:
                self.screen.fill(BLACK)
                self.terrain.draw(self.screen)

                # Draw all active landers, setting their color based on state
                for agent_data in active_agents:
                    lander_to_draw = agent_data['lander']
                    # Determine color based on final state or "best" status
                    if lander_to_draw.landed and self.terrain.is_in_landing_zone(lander_to_draw) and self._is_soft_landing(lander_to_draw):
                        lander_to_draw.color = GREEN # Successful landing in LZ
                    elif lander_to_draw.landed and not self.terrain.is_in_landing_zone(lander_to_draw) and self._is_soft_landing(lander_to_draw):
                        lander_to_draw.color = YELLOW # Soft landing outside LZ
                    elif lander_to_draw.crashed:
                        lander_to_draw.color = RED # Crashed
                    elif agent_data['genome'] is current_best_genome_in_gen: # Check if it's the current best flying agent
                        lander_to_draw.color = BLUE
                    else: # Default color for other flying agents
                        lander_to_draw.color = WHITE
                    lander_to_draw.draw(self.screen)

                # Display info
                info_text = self.font.render(
                    f"Gen: {self.neat_ga.generation + 1} Active Agents: {len(active_agents) - num_finished}/{len(active_agents)}",
                    True, WHITE
                )
                self.screen.blit(info_text, (10, 10))

                # Display status of the best agent (if available)
                if current_best_genome_in_gen: # Use current_best_genome_in_gen for status
                    best_lander_status = "N/A" # Initialize here, will be updated below
                    # Find the lander associated with the current_best_genome_in_gen
                    # Now, since current_best_genome_in_gen is a direct reference, this comparison works
                    for agent_data in active_agents:
                        if agent_data['genome'] is current_best_genome_in_gen: # Use 'is' for identity comparison
                            if lander_to_draw.landed and self.terrain.is_in_landing_zone(lander_to_draw) and self._is_soft_landing(lander_to_draw):
                                best_lander_status = "BEST LANDED (LZ)!"
                            elif lander_to_draw.landed and not self.terrain.is_in_landing_zone(lander_to_draw) and self._is_soft_landing(lander_to_draw):
                                best_lander_status = "BEST LANDED (OUTSIDE LZ)!"
                            elif agent_data['lander'].crashed:
                                best_lander_status = "BEST CRASHED!"
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

                pygame.display.flip()

        # After all agents are finished, store the successful landings count for the next generation's check
        self.previous_successful_landings_count = self.successful_landings_count_current_gen
        return [agent['fitness'] for agent in active_agents]


    def run_neat_evolution(self, generations):
        """
        Main loop to run the NEAT evolution with the Pygame simulation.
        """
        # Attempt to load a saved network at the very beginning
        loaded_genome = load_network("best_lander_nn.txt", self.neat_ga.input_size, self.neat_ga.output_size)
        if loaded_genome:
            print("Loaded existing neural network from best_lander_nn.txt")
            # Replace a random genome in the initial population with the loaded one, or add it.
            self.neat_ga.population[0] = loaded_genome # Replace the first one
            self.best_genome_for_display = loaded_genome.clone() # Set for initial display
            self.neat_ga._speciate() # Re-speciate with the loaded genome
        else:
            # If no network is loaded, initialize best_genome_for_display with the first genome
            # This ensures the overlay can be drawn from the start of the first generation.
            if self.neat_ga.population:
                self.best_genome_for_display = self.neat_ga.population[0].clone()


        gen = 0 # Initialize generation counter for unlimited loop
        while True: # Run for an unlimited number of generations
            # Update GA's generation attribute
            self.neat_ga.generation = gen

            # Conditional terrain generation:
            # Only generate new terrain if it's not the very first generation (gen > 0)
            # AND a majority (e.g., >= 50%) of agents successfully landed in the previous generation.
            if gen > 0 and self.previous_successful_landings_count >= self.neat_ga.population_size * 0.5:
                self.terrain.generate_terrain()
                if self.render_enabled:
                    print(f"--- New Generation {self.current_generation_number + 1}: Terrain Randomized! ---")
            else:
                if self.render_enabled:
                    print(f"--- Generation {self.current_generation_number + 1}: Keeping current terrain. ---")


            # Reset per-generation state for terrain randomization
            # self.successful_landings_count_current_gen is reset at the start of run_generation_simulation
            self.terrain_randomized_this_gen = False
            self.current_generation_number = gen # Update current generation number


            # Run simulation for the entire population and get all fitnesses
            fitness_results = self.run_generation_simulation(self.neat_ga.population)

            # Assign fitnesses back to genomes
            for i, genome in enumerate(self.neat_ga.population):
                genome.fitness = fitness_results[i]

            # Evolve the population based on collected fitnesses
            # Pass a dummy fitness function as fitness is already set by the simulation
            best_genome_this_gen = self.neat_ga.evolve(lambda g: g.fitness, generations=1)

            # Update the best genome for display (overall best)
            if not self.best_genome_for_display or best_genome_this_gen.fitness > self.best_genome_for_display.fitness:
                self.best_genome_for_display = best_genome_this_gen.clone()

            gen += 1 # Increment generation counter

        # Save the best genome on exit (this part will only be reached if sys.exit() is called)
        if self.best_genome_for_display:
            save_network(self.best_genome_for_display, "best_lander_nn.txt")
            print("Best neural network saved to best_lander_nn.txt")

        pygame.quit()

# --- Save/Load Functions ---
def save_network(genome, filename):
    """Saves a NeuralNetwork genome to a file."""
    try:
        with open(filename, 'w') as f:
            f.write(genome.to_string())
        return True
    except IOError as e:
        print(f"Error saving network to {filename}: {e}")
        return False

def load_network(filename, input_size, output_size):
    """Loads a NeuralNetwork genome from a file."""
    try:
        with open(filename, 'r') as f:
            s = f.read()
            loaded_nn = NeuralNetwork.from_string(s)
            # Basic validation to ensure loaded network matches expected sizes
            if loaded_nn and loaded_nn.input_size == input_size and loaded_nn.output_size == output_size:
                return loaded_nn
            else:
                print(f"Loaded network from {filename} has incompatible dimensions or is corrupted.")
                return None
    except IOError:
        print(f"No saved network found at {filename}. Starting fresh.")
        return None
    except Exception as e:
        print(f"Error loading network from {filename}: {e}")
        return None


# --- Main Execution ---
if __name__ == "__main__":
    # NEAT configuration
    neat_config = {
        'population_size': 100, # Increased from 50
        'add_node_prob': 0.05, # Increased from 0.03
        'add_connection_prob': 0.08, # Increased from 0.05
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
    #                terrain_height_at_lander_x, terrain_slope_at_lander_x
    NN_INPUT_SIZE = 11
    # Outputs for NN: main_thrust (0-1), rotation_thrust (-1 to 1)
    NN_OUTPUT_SIZE = 2

    # Initialize Genetic Algorithm
    neat_ga_instance = GeneticAlgorithm(
        population_size=neat_config['population_size'],
        input_size=NN_INPUT_SIZE,
        output_size=NN_OUTPUT_SIZE,
        config=neat_config
    )

    # Initialize and run the simulation
    # Set render_enabled to False for faster training, True to visualize
    simulation = LunarLanderSimulation(SCREEN_WIDTH, SCREEN_HEIGHT, neat_ga_instance, render_enabled=True)
    simulation.run_neat_evolution(generations=100) # The 'generations' argument is now effectively ignored for the main loop
