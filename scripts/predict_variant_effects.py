'''
This script differs from the default precompute_embeddings.py script in that it
computes embeddings for two sequences: the reference sequence and the variant
sequence. The variant sequence is obtained by replacing the reference nucleotide
with the variant nucleotide at the variant position.
'''
import argparse
from bend.utils import embedders, Annotation
from tqdm.auto import tqdm
from scipy import spatial
import time
import torch
import torchvision.models as models
from torch.profiler import profile, ProfilerActivity, record_function
import numpy as np


def main():

    parser = argparse.ArgumentParser('Compute embeddings')
    parser.add_argument('bed_file', type=str, help='Path to the bed file')
    parser.add_argument('out_file', type=str, help='Path to the output file')
    # model can be any of the ones supported by bend.utils.embedders
    parser.add_argument('model', choices=['ag', 'nt', 'dnabert', 'awdlstm', 'gpn', 'convnet', 'genalm', 'hyenadna', 'dnabert2','grover'], type=str, help='Model architecture for computing embeddings')
    parser.add_argument('checkpoint', type=str, help='Path to or name of the model checkpoint')
    parser.add_argument('genome', type=str, help='Path to the reference genome fasta file')
    parser.add_argument('--extra_context', type=int, default=128, help='Number of extra nucleotides to include on each side of the sequence')
    parser.add_argument('--kmer', type=int, default=3, help = 'Kmer size for the DNABERT model')
    parser.add_argument('--embedding_idx', type=int, default=-1, help = 'Index of the embedding to use for computing the distance')

    args = parser.parse_args()

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

    genome_annotation.annotation['distance'] = None

    Ncount = 0
    start = time.perf_counter()

    real_out_path = f"real_embeddings_{args.out_file}.csv"
    alt_out_path = f"alt_embeddings_{args.out_file}.csv"

    with open(real_out_path, "w+") as real_f, open(alt_out_path, "w+") as alt_f:
    #, \
    #profile(activities=[ProfilerActivity.CUDA], profile_memory=False) as prof:
    #    with record_function("model_inference"):
        for index, row in tqdm(genome_annotation.annotation.iterrows()):


            # middle_point = row['start'] + 256
            # index the right embedding with dna[len(dna)//2]
            dna = genome_annotation.get_dna_segment(index = index)

            #dna = dna.replace('N', 'A')
            #Ncount += dna.count('N')

            dna_alt = [x for x in dna]
            if extra_context_left == extra_context_right:
                dna_alt[len(dna_alt)//2] = row['alt']
            elif extra_context_right == 0:
                dna_alt[-1] = row['alt']
            elif extra_context_left == 0:
                dna_alt[0] = row['alt']
            else:
                raise ValueError('Not implemented')
            dna_alt = ''.join(dna_alt)

            embedding_wt, embedding_alt = embedder.embed([dna, dna_alt], **kwargs)
            embedding_wt = embedding_wt[0, args.embedding_idx]
            embedding_alt = embedding_alt[0, args.embedding_idx]

            #np.savetxt(real_f, embedding_wt.reshape(1, -1), delimiter=",")
            #np.savetxt(alt_f, embedding_alt.reshape(1, -1), delimiter=",")

            d = spatial.distance.cosine(embedding_alt, embedding_wt)
            genome_annotation.annotation.loc[index, 'distance'] = d


    #this is useful for alphagenome
    genome_annotation.annotation.to_csv(f"cosine_distance_{args.out_file}.csv")
    print(f"total Ns: {Ncount}")

    elapsed = time.perf_counter() - start
    print(f"Total elapsed time: {elapsed}")

    #avg = prof.key_averages()

    #total_self_device = sum(e.self_device_time_total for e in avg) / 1000  # ms
    #total_self_cpu  = sum(e.self_cpu_time_total for e in avg) / 1000   # ms
    #total_calls     = sum(e.count for e in avg)
    #total_self_device_mem = sum(e.self_device_memory_usage for e in avg)  # bytes
    #total_self_cpu_mem  = sum(e.self_cpu_memory_usage for e in avg)   # bytes

    #print(f"Total GPU compute time: {total_self_device:.1f} ms")
    #print(f"Total CPU time: {total_self_cpu:.1f} ms")
    #print(f"Total kernel/op calls: {total_calls}")
    #print(f"Total GPU memory allocated (self, all ops): {total_self_device_mem / 1e9:.3f} GB")
    #print(f"Total CPU memory allocated (self, all ops): {total_self_cpu_mem / 1e9:.3f} GB")



if __name__ == '__main__':
    main()