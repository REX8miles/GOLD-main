import matplotlib.pyplot as plt
import numpy as np

x_axis_data = [0.05, 0.075, 0.1, 0.125, 0.15]  # x
y_axis_data = [0.791952568, 0.784448004, 0.81, 0.773657079, 0.775709003]  # y

# Calculate the spacing between points on the X axis
spacing = np.diff(x_axis_data)

# Set up the figure and axes
fig, ax = plt.subplots()
ax.plot(x_axis_data, y_axis_data, color=(139/255, 107/255, 155/255), linestyle='--', marker='>', alpha=0.5, linewidth=3, label='lambda')

# Add the data point labels at the midpoint of each interval
for i, (x, y) in enumerate(zip(x_axis_data, y_axis_data)):
    ax.text(x, y + 0.005, f'{x}', ha='center', va='bottom', fontsize=7.5)
# Set the title and labels
ax.set_title('lambda experiment')
ax.set_xlabel('Hyper-parameter value')
ax.set_ylabel('Accuracy')

# Set the y-axis limits
ax.set_ylim(0.7, 0.85)

# Display the legend
ax.legend()

# Show the plot
plt.show()
# import matplotlib.pyplot as plt
# import numpy as np
#
# x_axis_data = [0.005, 0.0075, 0.01, 0.0125, 0.015]  # x
# y_axis_data = [0.900289961,	0.89808655,	0.905301127, 0.90123117,0.902263706]  # y
#
# # Calculate the spacing between points on the X axis
# spacing = np.diff(x_axis_data)
#
# # Set up the figure and axes
# fig, ax = plt.subplots()
# ax.plot(x_axis_data, y_axis_data, color=(189/255, 100/255, 38/255), linestyle='--', marker='>', alpha=0.5, linewidth=3, label='eta')
#
# # Add the data point labels at the midpoint of each interval
# for i, (x, y) in enumerate(zip(x_axis_data, y_axis_data)):
#     ax.text(x, y + 0.005, f'{x}', ha='center', va='bottom', fontsize=7.5)
# # Set the title and labels
# ax.set_title('eta experiment')
# ax.set_xlabel('Hyper-parameter value')
# ax.set_ylabel('Accuracy')
#
# # Set the y-axis limits
# ax.set_ylim(0.82, 0.92)
#
# # Display the legend
# ax.legend()
#
# # Show the plot
# plt.show()
