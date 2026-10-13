import pandas as pd
from Bio import SeqIO

v = pd.read_csv("data/variant_effects/mpra/mpra_variant_effects_filtered.bed", sep="\t")
#v = pd.read_csv("data/variant_effects/variant_effects_disease.bed", sep="\t")
v["start"] = v["start"].astype(int)
v["chromosome"] = v["chromosome"].str.replace("chr", "", regex=False).astype(str)

s = v.sample(2000, random_state=0)  # SNVs only



for name, path in [("hg19", "GRCh37.no-chr.fa"), ("hg38", "GRCh38.primary_assembly.genome.fa")]:
    genome_dict = SeqIO.to_dict(SeqIO.parse(f"data/genomes/{path}", "fasta"))

    hits = 0
    for chrom, pos, ref, alt in zip(s["chromosome"], s["start"], s["ref"], s["alt"]):

        if name == "hg38":
            chrom = "chr" + chrom

        hg_variant =  str(genome_dict[chrom].seq[int(pos)])
        if hg_variant.upper() == ref.upper():
            hits += 1

    print("Reference: ", name, " Hits: " , hits / len(s))