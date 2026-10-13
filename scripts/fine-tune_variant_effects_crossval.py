import argparse
import os
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.model_selection import train_test_split, StratifiedKFold


class VariantPairDataset(Dataset):
    def __init__(self, wt_sequences, alt_sequences, labels):
        assert len(wt_sequences) == len(alt_sequences) == len(labels)
        self.wt = wt_sequences
        self.alt = alt_sequences
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "wt": self.wt[idx],
            "alt": self.alt[idx],
            "label": self.labels[idx],
        }


def extract_embedding(embed_output, idx):
    return torch.stack([torch.as_tensor(e[idx]) for e in embed_output])


def contrastive_loss(emb_wt, emb_alt, labels, margin=10.0):
    labels = labels.float()
    cos_sim = F.cosine_similarity(emb_wt, emb_alt, dim=-1)
    distance = 1.0 - cos_sim  # cosine distance, range [0, 2]

    pathogenic_term = labels * torch.clamp(margin - distance, min=0.0) ** 2
    benign_term = (1.0 - labels) * distance ** 2

    return (pathogenic_term + benign_term).mean()


def build_loader(wt_sequences, alt_sequences, labels, batch_size, shuffle):
    dataset = VariantPairDataset(
        wt_sequences,
        alt_sequences,
        labels,
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def run_epoch(embedder, loader, device, embedding_idx, optimizer=None):
    """One pass over `loader`. Trains if optimizer is given, else evaluates."""
    train_mode = optimizer is not None
    embedder.model.train() if train_mode else embedder.model.eval()

    total_loss = 0.0
    context = torch.enable_grad() if train_mode else torch.no_grad()
    with context:
        for batch in loader:
            labels_batch = batch["label"].to(device)

            emb_wt = embedder.training_embed(
                batch["wt"], embedding_idx, use_grads=train_mode
            ).to(device)
            emb_alt = embedder.training_embed(
                batch["alt"], embedding_idx, use_grads=train_mode
            ).to(device)

            loss = contrastive_loss(emb_wt, emb_alt, labels_batch)

            if train_mode:
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            total_loss += loss.item()

    return total_loss / len(loader)


def train_model(embedder, train_loader, val_loader, device, args, save_path=None):
    """Trains for args.epochs, tracking + optionally saving the best val checkpoint.
    Returns the best val loss achieved."""
    optimizer = torch.optim.AdamW(
        [p for p in embedder.model.parameters() if p.requires_grad], lr=1e-5
    )

    best_val_loss = float("inf")
    for epoch in range(args.epochs):
        train_loss = run_epoch(embedder, train_loader, device, args.embedding_idx, optimizer)
        val_loss = run_epoch(embedder, val_loader, device, args.embedding_idx)

        print(f"  Epoch {epoch + 1}/{args.epochs}: train loss = {train_loss:.4f}, val loss = {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            if save_path is not None:
                embedder.model.save_pretrained(save_path)

    return best_val_loss


def get_embedder(args):
    kwargs = {'disable_tqdm': True}
    extra_context_left = args.extra_context
    extra_context_right = args.extra_context

    if args.model == 'ag':
        embedder = embedders.AlphaGenomeEmbedder(args.checkpoint)
    elif args.model == 'nt':
        embedder = embedders.NucleotideTransformerEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True
    elif args.model == 'dnabert':
        embedder = embedders.DNABertEmbedder(args.checkpoint, kmer=args.kmer)
    elif args.model == 'awdlstm':
        extra_context_right = 0
        embedder = embedders.AWDLSTMEmbedder(args.checkpoint)
    elif args.model == 'gpn':
        embedder = embedders.GPNEmbedder(args.checkpoint)
    elif args.model == 'convnet':
        embedder = embedders.ConvNetEmbedder(args.checkpoint)
    elif args.model == 'genalm':
        embedder = embedders.GENALMEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True
    elif args.model == 'hyenadna':
        embedder = embedders.HyenaDNAEmbedder(args.checkpoint)
        extra_context_right = 0
    elif args.model == 'dnabert2':
        embedder = embedders.DNABert2Embedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True
    elif args.model == 'grover':
        embedder = embedders.GROVEREmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True
    else:
        raise ValueError('Model not supported')

    return embedder, extra_context_left, extra_context_right


def main():
    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('model', choices=['ag', 'nt', 'dnabert', 'awdlstm', 'gpn', 'convnet', 'genalm', 'hyenadna', 'dnabert2', 'grover'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('batch_size', type=int, help='Fine-tuning batch size')
    parser.add_argument('epochs', type=int, help='Fine-tuning epochs per fold')
    parser.add_argument('--extra_context', type=int, default=256, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=3, help='Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=256, help='Index of the embedding to use for computing the distance')
    parser.add_argument('--n_folds', type=int, default=5, help='Number of cross-validation folds')
    parser.add_argument('--out_dir', type=str, default=None, help='Where fold/final checkpoints are saved')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #os.makedirs(args.out_dir, exist_ok=True)

    embedder, extra_context_left, extra_context_right = get_embedder(args)

    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)

    if args.extra_context > 0:
        genome_annotation.extend_segments(extra_context_left=extra_context_left, extra_context_right=extra_context_right)

    wt_sequences, alt_sequences, labels = [], [], []
    for index, row in tqdm(genome_annotation.annotation.iterrows(), desc="Building sequence pairs"):
        dna = genome_annotation.get_dna_segment(index=index)
        dna_alt = list(dna)
        dna_alt[len(dna_alt) // 2] = row['alt']
        dna_alt = ''.join(dna_alt)

        wt_sequences.append(dna)
        alt_sequences.append(dna_alt)
        labels.append(row['label'])

    # temp
    wt_sequences = wt_sequences[:100]
    alt_sequences = alt_sequences[:100]
    labels = labels[:100]

    # 2) 5-fold (stratified) cross-validation on the remaining trainval data.
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=42)

    fold_val_losses = []
    for fold, (tr_rel, val_rel) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        print(f"\n=== Fold {fold + 1}/{args.n_folds} ===")

        train_loader = build_loader(wt_sequences, alt_sequences, labels, args.batch_size, shuffle=True)
        val_loader = build_loader(wt_sequences, alt_sequences, labels, args.batch_size, shuffle=False)

        # Reset the embedder's trainable weights for each fold.
        embedder, _, _ = get_embedder(args)
        embedder.prepare_for_finetuning()

        #fold_ckpt = os.path.join(args.out_dir, f"fold_{fold + 1}")
        best_val_loss = train_model(embedder, train_loader, val_loader, device, args, save_path=None)
        fold_val_losses.append(best_val_loss)
        print(f"Fold {fold + 1} best val loss: {best_val_loss:.4f}")

    fold_val_losses = np.array(fold_val_losses)
    print(f"\nCV val loss: {fold_val_losses.mean():.4f} +/- {fold_val_losses.std():.4f}")



if __name__ == '__main__':
    main()