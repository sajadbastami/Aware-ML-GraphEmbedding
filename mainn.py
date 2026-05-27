import numpy as np
import networkx as nx
from utils import get_random_walks, simhash
from model import train_word2vec, build_cnn_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, hamming_loss

# 1. Generate Synthetic Multi-label Graph Data
def generate_data():
    G = nx.fast_gnp_random_graph(200, 0.05)
    labels = {i: np.random.randint(0, 2, size=5).tolist() for i in G.nodes()} # 5 possible labels
    return G, labels

def run_pipeline():
    print("Step 1: Data Preparation...")
    G, node_labels = generate_data()
    
    print("Step 2: Neighborhood Label Extraction & SimHash...")
    # Get walks to find neighbors
    walks = get_random_walks(G, num_walks=10, walk_length=10)
    
    # For each node, create a sequence of hashed neighborhood labels
    node_sequences = []
    for node in G.nodes():
        neighbors = list(G.neighbors(node))
        # Get labels of neighbors
        nbr_labels = [node_labels[nbr] for nbr in neighbors]
        # Flatten and Hash
        flat_labels = [str(l) for sublist in nbr_labels for l in sublist]
        h_val = simhash(flat_labels)
        node_sequences.append([h_val] * 10) # Simplified sequence for Word2Vec

    print("Step 3: Training Word2Vec (Feature-Aware Embedding)...")
    w2v_model = train_word2vec(node_sequences, vector_size=128)
    
    # Create feature matrix (X)
    X = []
    for seq in node_sequences:
        vecs = [w2v_model.wv[val] for val in seq]
        X.append(vecs)
    X = np.array(X) # Shape: (Nodes, Sequence_Length, Vector_Size)
    
    # Target (Y)
    Y = np.array([node_labels[i] for i in G.nodes()])

    print("Step 4: CNN Training & Evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
    
    cnn = build_cnn_model(input_shape=(X.shape[1], X.shape[2]), num_classes=Y.shape[1])
    cnn.fit(X_train, y_train, epochs=5, batch_size=16, verbose=1)
    
    # Predictions
    y_pred_prob = cnn.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)
    
    # Metrics
    micro_f1 = f1_score(y_test, y_pred, average='micro')
    h_loss = hamming_loss(y_test, y_pred)
    
    print(f"\n--- Results ---")
    print(f"Micro-F1 Score: {micro_f1:.4f}")
    print(f"Hamming Loss: {h_loss:.4f}")

if __name__ == "__main__":
    run_pipeline()