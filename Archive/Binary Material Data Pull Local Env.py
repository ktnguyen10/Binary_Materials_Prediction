from pymatgen.ext.matproj import MPRester
from pymatgen.core import Composition
from pymatgen.analysis.local_env import CrystalNN
import pandas as pd
import matplotlib.pyplot as plt

coornum_calc = CrystalNN()

mpr = MPRester("s7nGaCXnx6g4iuPoFT4")


# Query all binary 1 to 1 materials
# "**" is all binary compounds of 1 to 1 ratio
# Alternative: anon_formula = {'A': 1, 'B': 1}
anon_formula = {'A': 1, 'B': 1}
binary_data = mpr.query(criteria={"anonymous_formula": anon_formula, "icsd_ids": {"$gte": 0}},
                        properties=["task_id", "pretty_formula",
                                    "spacegroup.symbol", "spacegroup.crystal_system", "e_above_hull",
                                    "structure.lattice.volume", "density", "band_gap", "final_energy"])

# Change to Dataframe for ease of table manipulation
binary_df = pd.DataFrame(binary_data)

# Get the lowest energy above hull structure and place into separate dataframe
binary_df = binary_df.sort_values(by=['pretty_formula', 'e_above_hull'])

# Sort back by index
binary_df = binary_df.reset_index(drop=True)

# Visualization of Number of each structure in the data set
binaryRR_data = binary_df['spacegroup.crystal_system'].value_counts()
plot1 = binaryRR_data.plot(kind='bar')
plot1.set_xlabel('Crystal System')
plot1.set_ylabel('Count')

for rect in plot1.patches:
    y_value = rect.get_height()
    x_value = rect.get_x() + rect.get_width() / 7
    space = 1
    label = "{:.0f}".format(y_value)
    plot1.annotate(label, (x_value+0.2, y_value), xytext=(0, space),
                   textcoords="offset points", ha='center', va='bottom')

# Statistical Histograms for Numerical Data in Dataset
plt.figure()
hist1 = binary_df['e_above_hull'].hist(bins=100)
hist1.set_xlabel('Energy Above the Convex Hull [eV/atom]')
hist1.set_ylabel('Count')
hist1.set_xlim(0, 0.5)
plt.figure()
hist2 = binary_df['band_gap'].hist(bins=100)
hist2.set_xlabel('Band Gap [eV]')
hist2.set_ylabel('Count')
hist2.set_xlim(0, 2)
plt.figure()
hist3 = binary_df['density'].hist(bins=100)
hist3.set_xlabel('Density [g cm-3]')
hist3.set_ylabel('Count')


def get_cat_oxi_state(x):
    comp = Composition(x)
    oxi_states = comp.oxi_state_guesses()
    if len(oxi_states) > 0:
        oxi_states_output = list(oxi_states[0].values())
    else:
        oxi_states_output = [0, 0]

    return oxi_states_output[0]


def get_an_oxi_state(x):
    comp = Composition(x)
    oxi_states = comp.oxi_state_guesses()
    if len(oxi_states) > 0:
        oxi_states_output = list(oxi_states[0].values())
    else:
        oxi_states_output = [0, 0]

    return oxi_states_output[1]


# Local Env Approach
def get_cn(x):
    try:
        struct = mpr.get_structure_by_material_id(x)
        coornum = coornum_calc.get_cn(structure=struct, n=0)
    except:
        coornum = 0

    return coornum


binary_df['cat_oxi_state'] = binary_df["pretty_formula"].apply(get_cat_oxi_state)
binary_df['an_oxi_state'] = binary_df["pretty_formula"].apply(get_an_oxi_state)
# binary_df['coordination_num'] = binary_df["task_id"].apply(get_cn)

# binary_df_les.to_csv("Binary_Dataset_for_RRRule_localenv.csv")
