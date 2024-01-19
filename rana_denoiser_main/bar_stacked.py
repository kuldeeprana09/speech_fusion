import matplotlib.pyplot as plt
import numpy as np

# Data
groups = ['G1', 'G2', 'G3', 'G4', 'G5']
values1 = [12, 19, 14, 27, 16]
values2 = [21, 30, 15, 17, 20]

fig, ax = plt.subplots()

# Stacked bar chart
ax.bar(groups, values1, label = "Yes")
ax.bar(groups, values2, bottom = values1, label = "No")

# Sum of values
total_values = np.add(values1, values2)

# Total values labels
for i, total in enumerate(total_values):
  print("value of I",i)
  print("value of total", total)
  ax.text(i, total + 0.5, round(total),
          ha = 'center', weight = 'bold', color = 'black')

ax.legend()
ax.set_ylabel('Number of answers')

plt.show()