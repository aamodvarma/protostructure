import os
from tqdm import tqdm
from ase.io import read, write

# Folder containing your CIF files
cif_folder = "./lhs_cifs/"
# Output file
output_file = "combined_lhs.extxyz"

# Get a list of all CIF files
cif_files = [f for f in os.listdir(cif_folder) if f.endswith(".cif")]

# Initialize a list to store all structures
all_structures = []

for cif in tqdm(cif_files):
    filepath = os.path.join(cif_folder, cif)
    # Read the CIF file
    atoms_list = read(filepath, index=":")  # ':' ensures all frames if CIF has multiple
    if not isinstance(atoms_list, list):
        atoms_list = [atoms_list]
    all_structures.extend(atoms_list)

# Write all structures to a single EXTXYZ file
write(output_file, all_structures, format="extxyz")
print(f"Written {len(all_structures)} structures to {output_file}")