# import knihoven
import pandas as pd
import numpy as np


# nacteni dat
metabolome = pd.read_csv('./Data/hordeum_metabolites.csv', delimiter=';', index_col=None)
genome = pd.read_csv('./Data/hordeum_SNPs.csv', delimiter=';', index_col=None)

# uprava dataframu
metabolome.index = metabolome.iloc[:, 0].values
metabolome = metabolome.drop([metabolome.columns.values[0]], axis=1)

genome = genome.transpose()
genome.columns = genome.iloc[0]
genome = genome.iloc[1:, :]

# porovnani obsahu
matching_lines = np.intersect1d(genome.index, metabolome.index)
metabolome_ready = metabolome[metabolome.index.isin(matching_lines)]
genome_reduced = genome[genome.index.isin(matching_lines)]

# nahrazeni NaN hodnot nulou
genome_reduced.fillna(0, inplace=True)

# odstraneni prvniho sloupce genomického datasetu
# print(bar_genome_reduced.index)
# bar_genome_ready = bar_genome_reduced

# standardizace
genome_ready = genome_reduced.replace(2, -1)

# ulozeni predzpracovanych dat
# bar_metabolome_ready.to_csv('metabolome_preprocessed.csv')
# bar_genome_ready.to_csv('SNP_preprocessed.csv')

# printy
# print(metabolome)
# print(metabolome_ready)
# print(genome)
# print(genome_ready)
