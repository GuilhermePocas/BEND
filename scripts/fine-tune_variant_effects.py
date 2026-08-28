
import argparse
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial
import time
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

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


def contrastive_loss(emb_wt, emb_alt, labels, margin=1.0):

    labels = labels.float()
    cos_sim = F.cosine_similarity(emb_wt, emb_alt, dim=-1)
    distance = 1.0 - cos_sim  # cosine distance, range [0, 2]
 
    pathogenic_term = labels * torch.clamp(margin - distance, min=0.0) ** 2
    benign_term = (1.0 - labels) * distance ** 2
 
    return (pathogenic_term + benign_term).mean()

def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    # model can be any of the ones supported by bend.utils.embedders
    parser.add_argument('model', choices=['ag', 'nt', 'dnabert', 'awdlstm', 'gpn', 'convnet', 'genalm', 'hyenadna', 'dnabert2','grover'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('batch_size', type=int, help='Fine-tuning batch size')
    parser.add_argument('epochs', type=int, help='Fine-tuning epochs')
    parser.add_argument('--extra_context', type=int, default=256, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=3, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=256, help = 'Index of the embedding to use for computing the distance')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    extra_context_left = args.extra_context
    extra_context_right = args.extra_context

    kwargs = {'disable_tqdm': True}
    # get the embedder
    if args.model == 'ag':
         embedder = embedders.AlphaGenomeEmbedder(args.checkpoint)
    elif args.model == 'nt':
         embedder = embedders.NucleotideTransformerEmbedder(args.checkpoint)
         kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'dnabert':
        embedder = embedders.DNABertEmbedder(args.checkpoint, kmer = args.kmer)
    elif args.model == 'awdlstm':
        # autogressive model. No use for right context.
        extra_context_left = args.extra_context
        extra_context_right = 0
        embedder = embedders.AWDLSTMEmbedder(args.checkpoint)
    elif args.model == 'gpn':
        embedder = embedders.GPNEmbedder(args.checkpoint)
    elif args.model == 'convnet':
        embedder = embedders.ConvNetEmbedder(args.checkpoint)
    elif args.model == 'genalm':
        embedder = embedders.GENALMEmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'hyenadna':
        embedder = embedders.HyenaDNAEmbedder(args.checkpoint)
        # autogressive model. No use for right context.
        extra_context_left = args.extra_context
        extra_context_right = 0
    elif args.model == 'dnabert2':
        embedder = embedders.DNABert2Embedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    elif args.model == 'grover':
        embedder = embedders.GROVEREmbedder(args.checkpoint)
        kwargs['upsample_embeddings'] = True # each nucleotide has an embedding
    else:
        raise ValueError('Model not supported')
    

    # load the bed file
    genome_annotation = Annotation(args.bed_file, reference_genome=args.genome)


    # extend the segments if necessary
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

    dataset = VariantPairDataset(wt_sequences, alt_sequences, labels)
    train_loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True
    )

    optimizer = torch.optim.AdamW(
        [p for p in embedder.model.parameters() if p.requires_grad], lr=0.005
    )

    embedder.prepare_for_finetuning()

    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            print("caca")
            print(device)
            labels_batch = batch["label"].to(device)

            emb_wt = embedder.embed_with_grads(batch["wt"]).to(device)
            emb_alt = embedder.embed_with_grads(batch["alt"]).to(device)


            
            loss = contrastive_loss(emb_wt, emb_alt, labels_batch)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: avg loss = {epoch_loss / len(train_loader):.4f}")

    #embedder.save_finetuned(args.adapter_out)
    #print(f"Saved LoRA adapter to {args.adapter_out}")
    #print(f"Total elapsed time: {time.perf_counter() - start:.1f}s")


if __name__ == '__main__':
    main()