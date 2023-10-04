import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

data = np.array([
    [220,   0,   0,  10,   0,   0,  50,  17,   0,   0],
    [0, 280,   6,   0,   0,   0,   0,   0,   0,  11],
    [0,  15, 254,   1,   0,   0,   0,   0,   0,  27],
    [10,   1,   2, 256,   0,   0,  25,   0,   0,   3],
    [0,   0,   0,   0, 282,  10,   0,   1,   4,   0],
    [0,   0,   0,   0,  48, 200,   0,  21,  28,   0],
    [38,   0,   0,  17,   0,   0, 240,   2,   0,   0],
    [16,   0,   0,   9,   7,  26,  37, 186,  16,   0],
    [0,   0,   0,   0,  14,   8,   0,   5, 270,   0],
    [0,  29,  23,   0,   1,   0,   0,   0,   0, 244]])

classes = ['airport', 'bus', 'metro', 'metro_station', 'park', 'public_square', 'shopping_mall', 'street_pedestrian', 'street_traffic', 'tram']

# Calculate row sums to use for percentages
row_sums = data.sum(axis=1)

# Calculate percentages
percentages = (data.T / row_sums).T  # Transpose for division, then transpose back

# Format percentages as strings with '%' symbol
annot_data = [['{:.2f}'.format(val) for val in row] for row in percentages]
annot_kws = {'size': 6}
ax = sns.heatmap(data, xticklabels=classes, yticklabels=classes, annot=annot_data, fmt='', square=True, cmap='Blues', annot_kws=annot_kws)
ax.set_xlabel('Predicted')
ax.set_xticklabels(ax.get_xticklabels(), rotation = 70)
ax.set_ylabel('Actual')

plt.subplots_adjust(left=0.2, bottom=0.35)
plt.show()


