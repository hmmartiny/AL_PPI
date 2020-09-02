import pandas as pd

geneexp_cols = ['gene_expression_' + str(i+1) for i in range(0, 20)] # gene expression: 1-20
gomol_cols = ['GO_molecular_function_' + str(i+1) for i in range(0, (41-21)+1)] # GO molecular function: 21-41
gobio_cols = ['GO_biological_function_' + str(i+1) for i in range(0, (74-42)+1)] # GO biological function: 42-74
gocom_cols = ['GO_component_' + str(i+1) for i in range(0, (97-75)+1)] # GO component: 75-97
others = [
    'protein_expression', # 98
    'essentiality',  # 99
    'HMS_PCI_Mass',  # 100
    'TAP_Mass',  #101
    'Y2H',  #102
    'Synthetic_Lethal',  # 103
    'Gene_Co-occur', # 104
    'Sequence_Similarity' # 105
]
hm_pi_cols = ['Homology based PPI'.replace(' ', '_') + str(i+1) for i in range(0, (109-106)+1)] # Homology based PPI: 106-109
dom = ['Domain-Domain_Interaction'] # 110
prodna = ['Protein-DNA TF group binding_'.replace(' ', '_') + str(i+1) for i in range(0, (126-111)+1)] # Protein-DNA TF group binding: 111-126
mips_pc = ['MIPS_protein_class_' + str(i+1) for i in range(0, 151-127+1)] # MIPS protein class: 127-151
mips_mp = ['MIPS_mutant_phenotype_' + str(i+1) for i in range(0, (162-152+1))] # MIPS mutant phenotype: 151-162

input_cols = geneexp_cols + gomol_cols + gobio_cols + gocom_cols + others + hm_pi_cols + dom + prodna + mips_pc + mips_mp
output_cols = ['p1', 'p2', 'y']

all_cols = output_cols + input_cols + ['y2']

all_files = [
    'data/dipsPosPair',
    'data/dipsRandpairSub23w',
    'data/mipsPosPair',
    'data/mipsRandpairSub23w',
    'data/keggscie04PosPair',
    'data/keggscie04RandpairSub23w',
]

sets = []
for pf in all_files:
    df = pd.concat([
        pd.read_csv(pf, sep='\t', header=None),
        pd.read_csv(pf + '.feature', sep=',', header=None)
    ], axis=1)
    print(pf, df.shape)
    sets.append(df)
    
all_data = pd.concat(sets, axis=0)
all_data.columns = all_cols
all_data.drop_duplicates(['p1', 'p2'], inplace=True)
all_data.set_index(['p1', 'p2'])
print("all:", all_data.shape)

# reduce data features 
cols2reduce = {
    'GO_molecular_function': gomol_cols,
    'GO_biological_function': gobio_cols,
    'GO_localization': gocom_cols,
    'Homology_based_PPI': hm_pi_cols,
    'Protein-DNA_TF_group_binding': prodna,
    'MIPS_Protein_Class': mips_pc,
    'MIPS_Mutant_Phenotype': mips_mp,
}
for k in others:
    if k not in (['protein_expression', 'Sequence_Similarity']):
        cols2reduce[k.title()] = k

for k, cs in cols2reduce.items():
    try:
        all_data[k] = (all_data[cs] > -100).any(axis=1).astype('int')
    except ValueError:
        all_data[k] = (all_data[cs] > -100).any(axis=0).astype('int')
    except Exception as e:
        print(k, cs)
        raise e
    
reduced_cols = geneexp_cols + list(cols2reduce.keys()) + dom

all_data[reduced_cols].to_csv('data/protein_interaction_X_reduced.csv')
all_data['y'].to_csv('data/protein_interaction_y.csv')