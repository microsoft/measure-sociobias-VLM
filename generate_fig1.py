from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import json
import random

with open('avg_bertscore.json') as f:
    data = json.load(f)

root_dir = '/home/t-ashsathe/BlobStorage/containers/absathe/MultiModalBias/final_generations/humanoid_robot'
#occupations = list(filter(lambda x: len(x.split(' ')) == 1, [item['gold_occupation'].replace('/', ' or ') for item in data]))
occupations = list(filter(lambda x: 'All Other' not in x and len(x.split(',')) == 1, [item['gold_occupation'].replace('/', ' or ') for item in data]))
random.shuffle(occupations)
print(occupations, len(occupations))

final_occs = [
    'Telemarketers',
    'Radiologists',
    'Prosthodontists',
    'Machinists',
    'Barbers',
    'Clergy',
    'Cardiologists',
    'Models',

    'Cashiers',
    'Dancers',
    'Financial Examiners',
    'Paramedics',
    'Travel Agents',
    'Stonemasons',
    'Archivists',
    'School Bus Monitors',

    'Roofers',
    'Nannies',
    'Security Guards',
    'Veterinarians',
    'Flight Attendants',
    'Home Health Aides',
    'Firefighters',
    'Floral Designers',
]

final_occs += random.sample(set(occupations).difference(final_occs), 24 - len(final_occs))

# Generate some random images for demonstration
images = [Image.open(f'{root_dir}/{occ}.png') for occ in final_occs]
titles = final_occs
print(final_occs)
# Plotting the images
fig, axes = plt.subplots(3, 8, figsize=(16, 6))
#fig.suptitle('Examples of professions and corresponding images of the humanoid robot')

for ax, img, title in zip(axes.ravel(), images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.savefig('fig1.pdf', dpi=300, bbox_inches='tight')