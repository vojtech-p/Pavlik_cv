# import libraries
import pandas as pd
import numpy as np


# data load
metabolome = pd.read_csv('./Data/arabidopsis_metabolites.csv', delimiter=';', index_col=None)
metabolites_6C = metabolome[metabolome['condition'] == '6C'].drop(columns='condition')
metabolites_16C = metabolome[metabolome['condition'] == '16C'].drop(columns='condition')
metabolome = metabolome.drop(columns='condition')

SNPs = pd.read_csv('./Data/arabidopsis_SNPs.txt', delimiter='\t', index_col=None).T[3:]

# SNP dataset altering
SNPs.columns = SNPs.iloc[0]
SNPs.columns = [f"SNP_{i}" for i in range(SNPs.shape[1])]
SNPs['accession'] = SNPs.index.astype(np.int64)

# making the number of accessions in SNP dataset equal to the number of accession in metabolomic dataset
aligned_SNPs_6 = metabolites_6C[['accession']].merge(SNPs, on='accession', how='left')
aligned_SNPs_16 = metabolites_16C[['accession']].merge(SNPs, on='accession', how='left')
aligned_SNPs_all = metabolome[['accession']].merge(SNPs, on='accession', how='left')

# replacing NaN values with 0
aligned_SNPs_6 = aligned_SNPs_6.replace('.', 0).astype(np.int64)
aligned_SNPs_16 = aligned_SNPs_16.replace('.', 0).astype(np.int64)
aligned_SNPs_all = aligned_SNPs_all.replace('.', 0).astype(np.int64)

# preprocessed dataset save
# metabolites_6C.to_csv('ath_metabolome_6_preprocessed.csv')
# aligned_SNPs_6.to_csv('ath_SNP_6_preprocessed.csv')

# metabolites_16C.to_csv('ath_metabolome_16_preprocessed.csv')
# aligned_SNPs_16.to_csv('ath_SNP_16_preprocessed.csv')

# metabolome.to_csv('ath_metabolome_all_preprocessed.csv')
# aligned_SNPs_all.to_csv('ath_SNP_all_preprocessed.csv')

# prints
print(aligned_SNPs_all)
# print(aligned_SNPs)
# print(metabolites_6C)
# print(metabolites_6C)
