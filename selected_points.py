import matplotlib.pyplot as plt

selected_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata // 10) * 10
        y = int(event.ydata // 10) * 10
        if [x, y] not in selected_points and len(selected_points) < 2:
            selected_points.append([x, y])
            color = 'ro' if len(selected_points) == 1 else 'go'
            plt.plot(x, y, color, markersize=12)
            plt.draw()
        if len(selected_points) == 2:
            plt.title("Selected! Close window to continue.")
            plt.draw()
            fig.canvas.mpl_disconnect(cid)

# Draw grid
fig, ax = plt.subplots(figsize=(6,6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.set_xticks(range(0, 101, 10))
ax.set_yticks(range(0, 101, 10))
ax.grid(True)
plt.title("Click to select START (red) and TARGET (green)")

cid = fig.canvas.mpl_connect('button_press_event', onclick)
plt.show()

if len(selected_points) == 2:
    print(f"Start: {selected_points[0]}")
    print(f"Target: {selected_points[1]}")
    with open('DRL-based-path-finding/selected_points.txt', 'w') as f:
        f.write(f"{selected_points[0][0]},{selected_points[0][1]}\n")
        f.write(f"{selected_points[1][0]},{selected_points[1][1]}\n")
    print("Coordinates saved to DRL-based-path-finding/selected_points.txt")
else:
    print("Selection incomplete. Please run again.")
