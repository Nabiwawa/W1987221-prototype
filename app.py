# -----------------------------------------------------------------------------------
# PPML Demo: Upload/Built-in dataset + Technique selection
# Non-Private vs Different techniques (currently Differential Privacy) + Comparison
# -----------------------------------------------------------------------------------

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from opacus import PrivacyEngine
from opacus.accountants.analysis import rdp as rdp_analysis


# -----------------------------
# 0) Streamlit basic setup
# -----------------------------
st.set_page_config(page_title="PPML Demo", layout="wide")
st.title("Privacy-Preserving ML Demo")
st.markdown("""
    ## What this app does
    This tool demosntrates how Machine Learning behaves when privacy protection is applied.
    
    You can:
    - Train a normal ML model (no privacy)
    - Train a different privacy preserving ML model
    - Compare accuracy, loss and privacy trade-offs
            
    This helps you understand the balance between **model performance and data privacy**.
""")

st.write(
    "Upload a CSV or use the built-in synthetic dataset, choose a technique, "
    "then train and visualise results. Compare **Non-Private** vs **Privacy-Preserving Machine Learning Techniques**."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# 1) Sidebar: dataset + technique
# -----------------------------
st.sidebar.header("Dataset")
st.sidebar.info("""
    Dataset Options:
    - Synthetic dataset: Automatically generated data
    - CSV upload: Your own dataset (last column must be labeled)
                
    We use this to simulate real-world ML training scenarios.
""")

ds_choice = st.sidebar.selectbox(
    "Select data source",
    [
        "Select data source",
        "Use built-in synthetic dataset",
        "Upload CSV (last column = label)"
    ],
    index=0
)

uploaded_file = None
if ds_choice == "Upload CSV (last column = label)":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])

st.sidebar.markdown("---")
st.sidebar.header("Technique")
st.sidebar.info("""
    Tecniques Explained:
    
    - Baseline: Standard ML training (no privacy protection)
    - DIfferential Privacy: Adds noise to gradients to protect individual data
    - Federated Learning: Training across mulitple devices
    - Homomorphic Encryption: Computation on encrypted data
""")

technique = st.sidebar.selectbox(
    "Select technique",
    [
        "Select technique",
        "Non-Private (Baseline)",
        "Differential Privacy (DP-SGD)",
        "Federated Learning",
        "Homomorphic Encryption (Coming Soon)"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.header("Training Settings")
st.sidebar.info("""
    Training Settings Explained:
    
    - Epochs: How many times the model sees the dataset
    - Batch size: Number of samples processed at once
    - Learning Rate: How fast the model learns
""")
seed = st.sidebar.number_input("Random seed", min_value=0, value=42, step=1)
np.random.seed(seed); torch.manual_seed(seed)

epochs = st.sidebar.slider("Epochs", 1, 20, 8, step=1)
batch_size = st.sidebar.selectbox("Batch size", [32, 64, 128], index=1)
lr = st.sidebar.selectbox("Learning rate", [0.1, 0.05, 0.02, 0.01], index=3)

st.sidebar.markdown("---")
st.sidebar.subheader("DP (only used when DP is selected)")
st.sidebar.info("""
    Differential Privacy Controls:
    - Noise Multiplier: Higher = more privacy, lower accuracy
    - Max Grad Norm: Limits how much each data point can affect learning
    - Delta: Probability of privacy leakage (smaller = stronger privacy)
""")

noise_multiplier = st.sidebar.selectbox("Noise multiplier", [0.5, 0.8, 1.0, 1.2, 1.5], index=2)
max_grad_norm = st.sidebar.selectbox("Max grad norm", [0.5, 1.0, 1.5, 2.0], index=1)
delta_str = st.sidebar.text_input("Delta", value="1e-5")

st.sidebar.info(f"Device: **{device}**")

# -----------------------------
# 2) Load data helper (CSV or built-in)
# -----------------------------
def load_dataset(ds_choice, uploaded_file, seed):
    """
    Returns: X_train, x_test, y_train, y_test (numpy arrays)
    CSV format assumption: last column = label (binary).
    """
    if ds_choice == "Upload CSV (last column = label)":
        if uploaded_file is None:
            st.info("Please upload a CSV to proceed (last column must be the label).")
            st.stop()
        df = pd.read_csv(uploaded_file)
        if df.shape[1] < 2:
            st.error("CSV must have at least 2 columns (features + label).")
            st.stop()

        X = df.iloc[:, :-1].values
        y_raw = df.iloc[:, -1].values

        # Auto-map non 0/1 labels if exactly two unique values
        unique_vals = pd.unique(y_raw)
        if len(unique_vals) != 2:
            st.error("Label column must be binary (exactly two classes).")
            st.stop()
        # Map to {0,1} deterministically
        mapping = {unique_vals[0]: 0, unique_vals[1]: 1}
        y = np.vectorize(mapping.get)(y_raw)

        # Standardise features
        X = StandardScaler().fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        source = f"Uploaded CSV: {uploaded_file.name}"
    else:
        # Built-in synthetic dataset
        n_samples = 1200
        n_features = 16
        X, y = make_classification(
            n_samples=n_samples,
            n_features=n_features,
            n_informative=min(8, n_features),
            n_redundant=0,
            n_classes=2,
            random_state=seed
        )
        X = StandardScaler().fit_transform(X)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        source = "Built-in synthetic dataset"

    st.success(
        f"Dataset ready from **{source}** — Train: {len(y_train)}, Test: {len(y_test)}, Features: {X_train.shape[1]}"
    )
    return X_train, X_test, y_train, y_test



if ds_choice == "Select data source":
    st.warning("Please select a data source from the sidebar to continue.")
    st.stop()

X_train, X_test, y_train, y_test = load_dataset(ds_choice, uploaded_file, seed)


# To tensors + DataLoaders
def make_loaders(X_train, y_train, X_test, y_test, batch_size):
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)
    y_test_t  = torch.tensor(y_test, dtype=torch.long)

    train_ds = TensorDataset(X_train_t, y_train_t)
    test_ds  = TensorDataset(X_test_t, y_test_t)

    # Ensure batch size is valid
    bs = int(min(batch_size, max(2, len(train_ds))))
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, drop_last=True)
    test_loader  = DataLoader(test_ds, batch_size=bs, shuffle=False)
    return train_ds, train_loader, test_loader, bs

train_ds, train_loader_base, test_loader, batch_size = make_loaders(
    X_train, y_train, X_test, y_test, batch_size
)


# -----------------------------
# 3) Simple model + train/eval helpers
# -----------------------------
class TinyMLP(nn.Module):
    def __init__(self, in_features, hidden=32, out_features=2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_features)
        )

    def forward(self, x):
        return self.layers(x)


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total = 0.0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total += loss.item() * xb.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate_acc(model, loader, device):
    model.eval()
    preds, gts = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        pred = torch.argmax(logits, dim=1)
        preds.append(pred.cpu().numpy()); gts.append(yb.cpu().numpy())
    preds = np.concatenate(preds); gts = np.concatenate(gts)
    return accuracy_score(gts, preds)


def run_baseline(X_train, y_train, train_loader, test_loader, epochs, lr, device):
    model = TinyMLP(in_features=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    losses = []
    for _ in range(epochs):
        l = train_one_epoch(model, train_loader, crit, opt, device)
        losses.append(l)
    acc = evaluate_acc(model, test_loader, device)
    return model, losses, acc


def run_dp(X_train, y_train, train_ds, test_loader, epochs, lr, batch_size, device, sigma, max_gn):
    model = TinyMLP(in_features=X_train.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    # Opacus DataLoader (must match batch size + drop_last)
    train_loader_dp = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    privacy_engine = PrivacyEngine()
    model, opt, train_loader_dp = privacy_engine.make_private(
        module=model,
        optimizer=opt,
        data_loader=train_loader_dp,
        noise_multiplier=float(sigma),
        max_grad_norm=float(max_gn),
    )
    losses = []
    for _ in range(epochs):
        l = train_one_epoch(model, train_loader_dp, crit, opt, device)
        losses.append(l)
    acc = evaluate_acc(model, test_loader, device)
    return model, losses, acc, train_loader_dp

def run_federated(x_train, y_train, x_test, y_test, epochs, lr, device):

    num_clients = 3

    # Split data into clients
    X_splits = np.array_split(x_train, num_clients)
    y_splits = np.array_split(y_train, num_clients)

    
    client_data = [
        (
            torch.tensor(X_splits[i], dtype=torch.float32),
            torch.tensor(y_splits[i], dtype=torch.long)
        )
        for i in range(num_clients)
    ]

    global_model = TinyMLP(in_features=x_train.shape[1]).to(device)
    global_state = global_model.state_dict()

    losses = []

    for _ in range(epochs):
        client_states = []
        epoch_loss = 0

        for i in range(num_clients):
            model = TinyMLP(in_features=x_train.shape[1]).to(device)
            model.load_state_dict(global_state)

            opt = torch.optim.Adam(model.parameters(), lr=lr)
            crit = nn.CrossEntropyLoss()


            Xc, yc = client_data[i]

            loader = DataLoader(TensorDataset(Xc, yc), batch_size=32, shuffle=True)

            model.train()

            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = crit(model(xb), yb)
                loss.backward()
                opt.step()
                epoch_loss += loss.item() * xb.size(0)

            client_states.append(model.state_dict())

        
        new_state = {}
        for key in global_state.keys():
            new_state[key] = torch.stack(
                [cs[key] for cs in client_states], dim=0
            ).mean(dim=0)

        global_state = new_state
        global_model.load_state_dict(global_state)

        losses.append(epoch_loss / len(x_train))

    acc = evaluate_acc(
        global_model,
        DataLoader(
            TensorDataset(
                torch.tensor(x_test, dtype=torch.float32),
                torch.tensor(y_test, dtype=torch.long)
            ),
            batch_size=32
        ),
        device
    )

    return global_model, losses, acc
            


def estimate_epsilon(batch_size, train_ds_len, epochs, train_loader_dp, sigma, delta_str):
    try:
        delta = float(delta_str)
        sample_rate = batch_size / train_ds_len
        steps = epochs * len(train_loader_dp)
        orders = np.linspace(1.25, 64.0, 100)
        rdp = rdp_analysis.compute_rdp(q=sample_rate, noise_multiplier=float(sigma), steps=steps, orders=orders)
        eps, best_order = rdp_analysis.get_privacy_spent(orders=orders, rdp=rdp, delta=delta)
        return eps, best_order
    except Exception as e:
        return None, str(e)


# -----------------------------
# 4) Tabs: Single technique + Comparison
# -----------------------------
tab1, tab2 = st.tabs(["Run Selected Technique", "Full Comparison"])

# --- Tab 1: Run the selected technique ---
with tab1:
    st.header("Run Selected Technique")

    if technique == "Select technique":
        st.warning("Please select a technique from the sidebar to continue.")
        st.stop()

    if technique == "Non-Private (Baseline)":
        with st.spinner("Training Non-Private model..."):
            _, np_losses, np_acc = run_baseline(X_train, y_train, train_loader_base, test_loader, epochs, lr, device)
        st.success(f"Non-Private Test Accuracy: {np_acc:.3f}")

        st.info("""
        Result Explanation:
        - This model has no privacy protection
        - It usually performs better in accuracy 
        - However, it may risk exposing sensitive patterns in data
        """)

        fig_np, ax_np = plt.subplots()
        ax_np.plot(range(1, epochs + 1), np_losses, marker="o", label="Non-Private Loss")
        ax_np.set_xlabel("Epoch"); ax_np.set_ylabel("Loss"); ax_np.set_title("Non-Private Loss per Epoch")
        ax_np.legend()
        st.pyplot(fig_np)

    elif technique == "Differential Privacy (DP-SGD)":
        with st.spinner("Training DP model (Opacus)..."):
            _, dp_losses, dp_acc, train_loader_dp = run_dp(
                X_train, y_train, train_ds, test_loader, epochs, lr, batch_size, device,
                sigma=noise_multiplier, max_gn=max_grad_norm
            )
        st.success(f"DP Test Accuracy: {dp_acc:.3f}")

        st.info("""
        Result Explanation:
        - DP reduces accuracy due to added noise
        - This protects individual data points from being memorised by the model
        - Strong privacy usually means lower model performance
        """)

        fig_dp, ax_dp = plt.subplots()
        ax_dp.plot(range(1, epochs + 1), dp_losses, marker="o", color="orange", label="DP Loss")
        ax_dp.set_xlabel("Epoch"); ax_dp.set_ylabel("Loss"); ax_dp.set_title("DP Loss per Epoch")
        ax_dp.legend()
        st.pyplot(fig_dp)

        # Epsilon estimate (DP only)
        eps, best_order = estimate_epsilon(batch_size, len(train_ds), epochs, train_loader_dp, noise_multiplier, delta_str)
        if eps is not None:
            st.info(f"""
                Privacy Result (Differential Privacy):

                - Privacy strength (epsilon): {eps:.2f}
                - Risk level (delta): {delta_str}
                - Best internal setting used: {best_order:.2f}

                In simple terms:
                A LOWER epsilon means STRONGER privacy, but usually lower accuracy.
                A HIGHER epsilon means weaker privacy, but better model performance.
                """)
        else:
            st.warning(f"Could not estimate ε: {best_order}")

    elif technique == "Federated Learning":
        with st.spinner("Running Federated Learning simulation..."):
            fl_model, fl_losses, fl_acc = run_federated(X_train, y_train, X_test, y_test, epochs, lr, device)

        st.success(f"Federated Learning Accuracy: {fl_acc:.3f}")

        st.info("""
        Result Explanation: 
        - Data is split across multiple simulated clients
        - Each client trains locally
        - Models are averaged (FedAvg)
        - This preserves privacy by avoiding raw data sharing
        """)

        fig_fl, ax_fl = plt.subplots()
        ax_fl.plot(range(1, epochs + 1), fl_losses, marker="o", label="FL Loss")
        ax_fl.set_title("Federated Learning Loss per Epoch")
        ax_fl.legend()
        st.pyplot(fig_fl)

    elif technique == "Homomorphic Encryption (Coming Soon)":
        st.info("This feature has not been implemented yet. It will be added in a future update.")
        st.stop()
        
# --- Tab 2: Comparison ---
with tab2:
    st.header("Full Comparison")

    st.markdown("""
    ## What this comparison shows:
    This section highlights the trade-offs between:
    - Model accuracy (performance)
    - Privacy Protection (data safety)
    """)

    with st.spinner("Training all models for comparison..."):

        # Baseline
        _, np_losses, np_acc = run_baseline(
            X_train, y_train, train_loader_base, test_loader, epochs, lr, device
        )

        # DP
        _, dp_losses, dp_acc, train_loader_dp = run_dp(
            X_train, y_train, train_ds, test_loader, epochs, lr, batch_size, device,
            sigma=noise_multiplier, max_gn=max_grad_norm
        )

        # FL
        _, fl_losses, fl_acc = run_federated(
            X_train, y_train, X_test, y_test, epochs, lr, device
        )

    # --- Accuracy display ---
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Baseline")
        st.metric("Accuracy", f"{np_acc:.3f}")

    with col2:
        st.subheader("Differential Privacy")
        st.metric("Accuracy", f"{dp_acc:.3f}")

    with col3:
        st.subheader("Federated Learning")
        st.metric("Accuracy", f"{fl_acc:.3f}")

    # --- Loss comparison ---
    st.subheader("Loss Comparison")

    fig, ax = plt.subplots()
    ax.plot(range(1, epochs + 1), np_losses, label="Baseline")
    ax.plot(range(1, epochs + 1), dp_losses, label="DP")
    ax.plot(range(1, epochs + 1), fl_losses, label="FL")

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Loss Comparison Across Techniques")
    ax.legend()

    st.pyplot(fig)

    # --- DP epsilon ---
    eps, best_order = estimate_epsilon(batch_size, len(train_ds), epochs, train_loader_dp, noise_multiplier, delta_str)
    if eps is not None:
        st.info(f"""
            Privacy Result:

            - Privacy strength (epsilon): {eps:.2f}
            - Risk level (delta): {delta_str}
            - Best internal setting used: {best_order:.2f}

            In simple terms:
            A LOWER epsilon means STRONGER privacy, but usually lower accuracy.
            A HIGHER epsilon means weaker privacy, but better model performance.
            """)
    
    # --- Explanation ---
    st.info("""
    Comparison Insights:

    - Baseline: Highest accuracy but no privacy protection
    - Differential Privacy: Strong privacy, reduced accuracy due to noise
    - Federated Learning: Good balance, data stays local

    This demonstrates the trade-off between performance and privacy.
    """)



