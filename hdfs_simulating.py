import random

# Simulating HDFS Storage
data_nodes = {
    "DataNode 1": [],
    "DataNode 2": [],
    "DataNode 3": [],
    "DataNode 4": [],
    "DataNode 5": [],
    "DataNode 6": [],
}

# Sample data blocks to distribute
data_blocks = [
    "Transaction 1: Deposit $100",
    "Transaction 2: Withdraw $50",
    "Transaction 3: Transfer $200",
    "Transaction 4: Deposit $300",
    "Transaction 5: Withdraw $100"
    "Transaction 6: Deposit $100",
    "Transaction 7: Withdraw $50",
    "Transaction 8: Transfer $200",
    "Transaction 9: Deposit $300",
    "Transaction 10: Withdraw $100"
]

# Initial distribution of data blocks
for block in data_blocks:
    node = random.choice(list(data_nodes.keys()))
    data_nodes[node].append(block)

# Display the distributed data
print("Distributed Data:")
for node, blocks in data_nodes.items():  # Iterate over each DataNode
    print(f"{node}: {blocks}")  # Print the node name and the list of blocks it holds

# Simulation of data replication
replication_factor = 3  # Defines how many copies we want; in HDFS, the default is usually 3

print("\nSimulating data replication...")
for node in data_nodes:  # Iterate over each DataNode
    for block in data_nodes[node]:  # Iterate over each block in the current DataNode
        if random.random() < 0.5:  # Generates a random floating-point number between 0.0 and 1.0
            # Choose a different node for replication
            target_node = random.choice([n for n in data_nodes if n != node])
            if block not in data_nodes[target_node]:  # Append block only if it's not already present
                data_nodes[target_node].append(block)

# Display the data after replication
print("\nData distribution after replication:")
for node, blocks in data_nodes.items():  # Iterate over each DataNode
    print(f"{node}: {blocks}")  # Print the node name and the list of blocks it holds

# Simulation of Data Retrieval
def retrieve_transaction(transaction): # define a function that takes transaction as input
    """
    Simulate the retrieval of a transaction.
    """
    for node, blocks in data_nodes.items():  # Iterate over each DataNode and its associated blocks
        if transaction in blocks:  # Check if the specified transaction is present in the current DataNode
            return f"{transaction} found in {node}"  # Return a success message with the node name
    return f"{transaction} not found in any DataNode."  # Return failure message if not found

# Display the data after retrieval
print("\nSimulating transaction retrieval...")
print(retrieve_transaction("Transaction 1: Deposit $100"))  # Example transaction
print(retrieve_transaction("Transaction 2: Withdraw $50"))  # Example transaction
print(retrieve_transaction("Transaction 2: Deposit $50"))  # Example transaction
print(retrieve_transaction("Transaction 6: Nonexistent Transaction"))  # Nonexistent transaction

# Final diagnostic output
print("\nReplication and retrieval simulation complete.")
