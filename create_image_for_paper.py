import matplotlib.pyplot as plt
import textwrap as twp
import numpy as np
from mpl_toolkits.axes_grid1 import ImageGrid
from PIL import Image
root_dir = '/home/t-assathe/BlobStorage/modular-rai-gcr-blob-abs/MultiModalBias/dall-e-3/'
files_list = f'{root_dir}/../paper_anecdotes.txt'

image_list = [name.strip().replace('..', '.') for name in open(files_list).readlines()]

"""
fig = plt.figure(figsize=(15, 5))
grid = ImageGrid(fig, 111, nrows_ncols=(3, 8), axes_pad=0.05)
for ax, image_path in zip(grid, image_list):
    ax.imshow(np.asarray(Image.open(f'{root_dir}/{image_path}').convert('RGB')))
    ax.set_axis_off()
plt.savefig('image.pdf', dpi=300, bbox_inches='tight')
"""
fig, ax = plt.subplots(nrows=2, ncols=8, figsize=(15, 2.5))
plt.subplots_adjust(wspace=-0.1, hspace=-0.15)
for i, image_path in enumerate(image_list[:8]):
    ax[0][i].imshow(np.asarray(Image.open(f'{root_dir}/{image_path}').convert('RGB')))
    ax[0][i].set_axis_off()
    ax[1][i].set_xlim((0, 1))
    ax[1][i].set_ylim((0, 1))
    ax[1][i].text(0.5, 0.5, twp.fill(image_path.lower().replace('.png', '').replace('a blue humanoid robot is', ''), 16), verticalalignment='center', horizontalalignment='center')
    ax[1][i].set_axis_off()
    #ax[1][i].text(0.5, 0.5, 'hello', wrap=True, verticalalignment='center', horizontalalignment='center')
plt.savefig('image.pdf', dpi=300, bbox_inches='tight')