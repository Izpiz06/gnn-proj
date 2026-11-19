"""
scripts/run_ablation.py

Performs an ablation study comparing three architectures on the exact same
held-out test split of Chromosome 22:
1. Sequence-Only Baseline: DNABERT-2 with an MLP head (no Hi-C graph).
2. Graph-Only Baseline: 2-layer GAT on Hi-C contact graph (no DNA sequence).
3. Hybrid Model: DNABERT-2 + 2-layer GAT with aligned modality fusion.

Outputs:
- ablation_results.json: Raw metrics for all three models.
- figures/ablation_comparison.png: Publication-ready 3-panel comparative figure.
"""

import os
import time
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GATConv
from transformers import AutoModel
from transformers.models.bert.configuration_bert import BertConfig
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, f1_score, precision_score, recall_score, confusion_matrix
import bitsandbytes as bnb
from torch.cuda.amp import autocast, GradScaler
import torch.utils.checkpoint as checkpoint

# ==========================================
# 1. MODEL DEFINITIONS
# ==========================================
class SequenceOnlyModel(nn.Module):
    """Sequence-only baseline: DNABERT-2 with an MLP head (no Hi-C graph / GAT)."""
    def __init__(self, hidden_dim=768, num_classes=2, use_gradient_checkpointing=True):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        config.attention_probs_dropout_prob = 0.1
        self.dnabert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        self.bert_micro_batch = 4
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, blocks=None):
        if blocks is not None:
            num_seeds = blocks[-1].num_dst_nodes()
            input_ids = input_ids[:num_seeds]
            attention_mask = attention_mask[:num_seeds]
            
        micro_batch = self.bert_micro_batch
        all_embeddings = []
        for mb_i in range(0, input_ids.size(0), micro_batch):
            ids_chunk = input_ids[mb_i:mb_i+micro_batch]
            mask_chunk = attention_mask[mb_i:mb_i+micro_batch]
            if self.use_gradient_checkpointing and self.training:
                def custom_forward(ids, mask):
                    return self.dnabert(input_ids=ids, attention_mask=mask)[0]
                out = checkpoint.checkpoint(custom_forward, ids_chunk, mask_chunk, use_reentrant=False)
            else:
                out = self.dnabert(input_ids=ids_chunk, attention_mask=mask_chunk)[0]
            all_embeddings.append(out.mean(dim=1))
        seq_embeddings = torch.cat(all_embeddings, dim=0)
        return self.classifier(seq_embeddings)

class GraphOnlyModel(nn.Module):
    """Structure-only baseline: 2-layer GAT on Hi-C graph with learned node embeddings (no DNA sequence)."""
    def __init__(self, num_nodes, hidden_dim=768, num_classes=2):
        super().__init__()
        self.node_embeddings = nn.Embedding(num_nodes, hidden_dim)
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, num_heads=4, feat_drop=0.1, attn_drop=0.1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, num_heads=1, feat_drop=0.1, attn_drop=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, blocks):
        src_nids = blocks[0].srcdata[dgl.NID]
        h = self.node_embeddings(src_nids)
        h = self.gat1(blocks[0], h).flatten(1)
        h = F.elu(h)
        h = self.gat2(blocks[1], h).squeeze(1)
        return self.classifier(h)

class DNABertGNN(nn.Module):
    """Hybrid model: DNABERT-2 sequence embeddings fused with 2-layer GAT structural embeddings."""
    def __init__(self, hidden_dim=768, num_classes=2, use_gradient_checkpointing=True):
        super().__init__()
        self.use_gradient_checkpointing = use_gradient_checkpointing
        config = BertConfig.from_pretrained("zhihan1996/DNABERT-2-117M")
        config.attention_probs_dropout_prob = 0.1
        self.dnabert = AutoModel.from_pretrained("zhihan1996/DNABERT-2-117M", trust_remote_code=True, config=config)
        self.bert_micro_batch = 4
        self.gat1 = GATConv(hidden_dim, hidden_dim // 4, num_heads=4, feat_drop=0.1, attn_drop=0.1)
        self.gat2 = GATConv(hidden_dim, hidden_dim, num_heads=1, feat_drop=0.1, attn_drop=0.1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, input_ids, attention_mask, blocks):
        micro_batch = self.bert_micro_batch
        all_embeddings = []
        for mb_i in range(0, input_ids.size(0), micro_batch):
            ids_chunk = input_ids[mb_i:mb_i+micro_batch]
            mask_chunk = attention_mask[mb_i:mb_i+micro_batch]
            if self.use_gradient_checkpointing and self.training:
                def custom_forward(ids, mask):
                    return self.dnabert(input_ids=ids, attention_mask=mask)[0]
                out = checkpoint.checkpoint(custom_forward, ids_chunk, mask_chunk, use_reentrant=False)
            else:
                out = self.dnabert(input_ids=ids_chunk, attention_mask=mask_chunk)[0]
            all_embeddings.append(out.mean(dim=1))
        seq_embeddings = torch.cat(all_embeddings, dim=0)

        h = seq_embeddings
        h = self.gat1(blocks[0], h).flatten(1)
        h = F.elu(h)
        h = self.gat2(blocks[1], h).squeeze(1)

        num_seed = blocks[-1].num_dst_nodes()
        seed_seq = seq_embeddings[:num_seed]
        fused = torch.cat([seed_seq, h], dim=1)
        return self.classifier(fused)

# ==========================================
# 2. TRAINING & EVALUATION UTILITY
# ==========================================
def train_and_evaluate_model(model_name, model, train_loader, val_loader, test_loader, device, num_epochs=3, max_train_steps=30):
    print(f"\n==========================================")
    print(f"Training Architecture: {model_name} on {device}")
    print(f"==========================================")
    
    try:
        model = model.to(device)
    except Exception as e:
        print(f"Failed to place {model_name} on {device}: {e}. Falling back to CPU...")
        device = torch.device('cpu')
        model = model.to(device)

    is_cuda = (device.type == 'cuda')
    if is_cuda and ("Sequence" in model_name or "Hybrid" in model_name):
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=2e-5)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if "Graph" in model_name else 2e-5)
        
    accum_steps = 8
    scaler = GradScaler(enabled=is_cuda)
    history = {'epoch': [], 'train_loss': [], 'val_loss': [], 'val_acc': []}

    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            steps = 0
            for step, (in_nodes, out_nodes, blocks) in enumerate(train_loader):
                in_ids = blocks[0].srcdata['input_ids'].to(device)
                mask = blocks[0].srcdata['attention_mask'].to(device)
                labels = blocks[-1].dstdata['label'].to(device)
                blocks = [b.to(device) for b in blocks]

                with autocast(enabled=is_cuda, dtype=torch.float16):
                    logits = model(in_ids, mask, blocks)
                    loss = F.cross_entropy(logits, labels) / accum_steps

                scaler.scale(loss).backward()
                if (step + 1) % accum_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                epoch_loss += loss.item() * accum_steps
                steps += 1
                if steps >= max_train_steps:
                    break

            avg_train_loss = epoch_loss / max(steps, 1)

            # Validation
            model.eval()
            val_loss, val_corr, val_tot = 0.0, 0, 0
            with torch.no_grad():
                for v_step, (_, _, v_blocks) in enumerate(val_loader):
                    v_in_ids = v_blocks[0].srcdata['input_ids'].to(device)
                    v_mask = v_blocks[0].srcdata['attention_mask'].to(device)
                    v_labels = v_blocks[-1].dstdata['label'].to(device)
                    v_blocks = [b.to(device) for b in v_blocks]
                    with autocast(enabled=is_cuda, dtype=torch.float16):
                        v_logits = model(v_in_ids, v_mask, v_blocks)
                        v_l = F.cross_entropy(v_logits, v_labels)
                    val_loss += v_l.item()
                    preds = v_logits.argmax(dim=-1)
                    val_corr += (preds == v_labels).sum().item()
                    val_tot += v_labels.size(0)

            avg_val_loss = val_loss / max(v_step + 1, 1)
            val_acc = val_corr / max(val_tot, 1)

            history['epoch'].append(epoch + 1)
            history['train_loss'].append(avg_train_loss)
            history['val_loss'].append(avg_val_loss)
            history['val_acc'].append(val_acc)
            print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.2%}")

    except torch.cuda.OutOfMemoryError as oom_err:
        print(f"CUDA OutOfMemory detected during {model_name}. Rerunning on CPU...")
        del model, optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Fallback to CPU for this model
        if "Sequence" in model_name:
            m = SequenceOnlyModel()
        elif "Graph" in model_name:
            m = GraphOnlyModel(num_nodes=blocks[0].num_src_nodes())
        else:
            m = DNABertGNN()
        return train_and_evaluate_model(model_name, m, train_loader, val_loader, test_loader, torch.device('cpu'), num_epochs, max_train_steps)


    # Held-Out Test Evaluation
    print(f"\nEvaluating {model_name} on Held-Out Test Set (109 bins)...")
    model.eval()
    test_loss, test_preds, test_targets, test_probs = 0.0, [], [], []
    with torch.no_grad():
        for t_step, (_, _, t_blocks) in enumerate(test_loader):
            t_in_ids = t_blocks[0].srcdata['input_ids'].to(device)
            t_mask = t_blocks[0].srcdata['attention_mask'].to(device)
            t_labels = t_blocks[-1].dstdata['label'].to(device)
            t_blocks = [b.to(device) for b in t_blocks]
            with autocast(dtype=torch.float16):
                t_logits = model(t_in_ids, t_mask, t_blocks)
                t_l = F.cross_entropy(t_logits, t_labels)
            test_loss += t_l.item()
            probs = F.softmax(t_logits, dim=-1)[:, 1]
            preds = t_logits.argmax(dim=-1)
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(t_labels.cpu().numpy())

    test_preds = np.array(test_preds)
    test_targets = np.array(test_targets)
    test_probs = np.array(test_probs)

    test_acc = float(np.mean(test_preds == test_targets))
    auc_score = float(roc_auc_score(test_targets, test_probs))
    f1_macro = float(f1_score(test_targets, test_preds, average='macro'))
    f1_weighted = float(f1_score(test_targets, test_preds, average='weighted'))
    prec = float(precision_score(test_targets, test_preds, average='weighted', zero_division=0))
    rec = float(recall_score(test_targets, test_preds, average='weighted', zero_division=0))
    fpr, tpr, _ = roc_curve(test_targets, test_probs)

    results = {
        'model_name': model_name,
        'history': history,
        'test_loss': test_loss / max(t_step + 1, 1),
        'test_acc': test_acc,
        'roc_auc': auc_score,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': prec,
        'recall': rec,
        'fpr': fpr.tolist(),
        'tpr': tpr.tolist()
    }
    print(f"Results for {model_name}: Accuracy = {test_acc:.2%}, ROC-AUC = {auc_score:.4f}, F1-Macro = {f1_macro:.4f}")
    
    # Clean up GPU memory before next model
    del model, optimizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    return results

# ==========================================
# 3. PLOTTING COMPARATIVE FIGURES
# ==========================================
def plot_ablation_results(all_results, output_path="figures/ablation_comparison.png"):
    print(f"\nGenerating comparative ablation figure at {output_path}...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {
        'Sequence-Only Baseline': '#1f77b4',
        'Graph-Only Baseline': '#ff7f0e',
        'Hybrid Model (Ours)': '#2ca02c'
    }

    # 1. Comparative ROC Curves
    for r in all_results:
        name = r['model_name']
        axes[0].plot(r['fpr'], r['tpr'], label=f"{name} (AUC = {r['roc_auc']:.3f})", color=colors[name], linewidth=2.2)
    axes[0].plot([0, 1], [0, 1], 'k--', linewidth=1)
    axes[0].set_title('Test ROC Curves Comparison', fontweight='bold', fontsize=12)
    axes[0].set_xlabel('False Positive Rate', fontsize=10)
    axes[0].set_ylabel('True Positive Rate', fontsize=10)
    axes[0].legend(loc='lower right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # 2. Performance Metrics Bar Chart
    model_names = [r['model_name'] for r in all_results]
    short_names = ['Sequence-Only', 'Graph-Only', 'Hybrid (Ours)']
    metrics_acc = [r['test_acc'] * 100 for r in all_results]
    metrics_auc = [r['roc_auc'] * 100 for r in all_results]
    metrics_f1 = [r['f1_weighted'] * 100 for r in all_results]

    x = np.arange(len(short_names))
    width = 0.25

    rects1 = axes[1].bar(x - width, metrics_acc, width, label='Accuracy (%)', color='#4c72b0')
    rects2 = axes[1].bar(x, metrics_auc, width, label='ROC-AUC (%)', color='#55a868')
    rects3 = axes[1].bar(x + width, metrics_f1, width, label='Weighted F1 (%)', color='#c44e52')

    axes[1].set_title('Test Performance by Architecture', fontweight='bold', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short_names, fontsize=10)
    axes[1].set_ylabel('Score (%)', fontsize=10)
    axes[1].set_ylim(40, 100)
    axes[1].legend(loc='lower right', fontsize=9)
    axes[1].grid(True, alpha=0.3, axis='y')

    # Add values above bars
    for rects in [rects1, rects2, rects3]:
        for rect in rects:
            height = rect.get_height()
            axes[1].annotate(f'{height:.1f}%',
                             xy=(rect.get_x() + rect.get_width() / 2, height),
                             xytext=(0, 3), textcoords="offset points",
                             ha='center', va='bottom', fontsize=7.5, fontweight='bold')

    # 3. Validation Loss Convergence
    for r in all_results:
        name = r['model_name']
        axes[2].plot(r['history']['epoch'], r['history']['val_loss'], marker='o', label=name, color=colors[name], linewidth=2)
    axes[2].set_title('Validation Loss Convergence', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Epoch', fontsize=10)
    axes[2].set_ylabel('Cross-Entropy Loss', fontsize=10)
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved: {output_path}")

# ==========================================
# 4. MAIN RUNNER
# ==========================================
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load Graph
    graph_path = "data/graph_chr22_50000bp.bin"
    if not os.path.exists(graph_path):
        raise FileNotFoundError(f"Graph cache not found at {graph_path}. Please run 3050.ipynb or build the graph first.")

    graphs, _ = dgl.load_graphs(graph_path)
    g = graphs[0]
    num_nodes = g.num_nodes()

    sampler = dgl.dataloading.MultiLayerNeighborSampler([2, 2])
    train_nids = torch.where(g.ndata['train_mask'])[0]
    val_nids = torch.where(g.ndata['val_mask'])[0]
    test_nids = torch.where(g.ndata['test_mask'])[0]

    train_loader = dgl.dataloading.DataLoader(g, train_nids, sampler, batch_size=2, shuffle=True, drop_last=False, num_workers=0)
    val_loader = dgl.dataloading.DataLoader(g, val_nids, sampler, batch_size=4, shuffle=False, drop_last=False, num_workers=0)
    test_loader = dgl.dataloading.DataLoader(g, test_nids, sampler, batch_size=4, shuffle=False, drop_last=False, num_workers=0)

    # 1. Sequence-Only Baseline
    seq_model = SequenceOnlyModel()
    seq_results = train_and_evaluate_model("Sequence-Only Baseline", seq_model, train_loader, val_loader, test_loader, device, num_epochs=3, max_train_steps=25)

    # 2. Graph-Only Baseline
    graph_model = GraphOnlyModel(num_nodes=num_nodes)
    graph_results = train_and_evaluate_model("Graph-Only Baseline", graph_model, train_loader, val_loader, test_loader, device, num_epochs=3, max_train_steps=25)

    # 3. Hybrid Multimodal Model
    hybrid_model = DNABertGNN()
    hybrid_results = train_and_evaluate_model("Hybrid Model (Ours)", hybrid_model, train_loader, val_loader, test_loader, device, num_epochs=3, max_train_steps=25)

    all_results = [seq_results, graph_results, hybrid_results]

    # Save Results
    with open("ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nSaved all metrics to ablation_results.json")

    # Generate Comparative Figure
    plot_ablation_results(all_results, output_path="figures/ablation_comparison.png")

    # Print Summary Table
    print("\n" + "="*85)
    print("ABLATION STUDY SUMMARY (Held-Out Test Set, 109 Bins)")
    print("="*85)
    print(f"{'Architecture':<26} | {'Accuracy':<10} | {'ROC-AUC':<10} | {'Weighted F1':<12} | {'Test Loss':<10}")
    print("-"*85)
    for r in all_results:
        print(f"{r['model_name']:<26} | {r['test_acc']*100:>8.2f}% | {r['roc_auc']:>10.4f} | {r['f1_weighted']*100:>10.2f}% | {r['test_loss']:>10.4f}")
    print("="*85)

if __name__ == "__main__":
    main()
