import matplotlib.pyplot as plt
import pandas as pd
from pymatgen.core import Composition

# Load dataset and filter garnet room temperature samples
df = pd.read_csv("LiIonDatabase.csv", skiprows=3)
garnet = df[df['family'] == 'Garnet']
garnet_rt = garnet[garnet['temperature'].between(15, 35)].copy().reset_index(drop=True)

# Visualize dopant distribution
dopants = ['Ta', 'Ba', 'Y', 'Al', 'Nb', 'Sr', 'Te', 'Ga']
counts = [garnet_rt['composition'].str.contains(d).sum() for d in dopants]

plt.figure(figsize=(8, 4))
plt.bar(dopants, counts, color='steelblue')
plt.xlabel('Dopant')
plt.ylabel('Sample Count')
plt.title('Dopant Distribution in Garnet Dataset (15-35°C)')
plt.tight_layout()
plt.savefig("figures/dopant_distribution.png", dpi=150)
plt.close()

print("Saved.")