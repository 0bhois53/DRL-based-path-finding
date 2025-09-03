import matplotlib.pyplot as plt
import os

selected_points = []

def onclick(event):
    if event.xdata is not None and event.ydata is not None:
        x = int(event.xdata // 10) * 10
        y = int(event.ydata // 10) * 10
        if [x, y] not in selected_points and len(selected_points) < 2:
            selected_points.append([x, y])
            color = 'ro' if len(selected_points) == 1 else 'go'
            plt.plot(x, y, color, markersize=12)
            label = "START" if len(selected_points) == 1 else "GOAL"
            plt.text(x+2, y+2, label, fontsize=12, fontweight='bold')
            plt.draw()
        if len(selected_points) == 2:
            plt.title("Selected! Close window to save points.")
            plt.draw()
            fig.canvas.mpl_disconnect(cid)

def setup_point_selection():
    """Setup the interactive point selection interface"""
    global fig, cid
    
    print("Custom Point Selection Tool")
    print("="*40)
    print("Instructions:")
    print("1. Click to select START point (red)")
    print("2. Click to select GOAL point (green)")
    print("3. Close the window when done")
    print("4. Points will be saved to selected_points.txt")
    print()
    
    # Check if file already exists
    if os.path.exists('selected_points.txt'):
        print("Current selected_points.txt content:")
        with open('selected_points.txt', 'r') as f:
            lines = f.read().strip().split('\n')
            for i, line in enumerate(lines):
                point_type = "START" if i == 0 else "GOAL"
                print(f"  {point_type}: {line}")
        print()
        
        overwrite = input("File exists. Overwrite with new points? (y/n): ").lower().strip()
        if overwrite not in ['y', 'yes', '1', 'true']:
            print("Keeping existing points. Exiting.")
            return False
    
    # Draw grid
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xticks(range(0, 101, 10))
    ax.set_yticks(range(0, 101, 10))
    ax.grid(True, alpha=0.3)
    plt.title("Click to select START (red) and GOAL (green) points")
    plt.xlabel("X Coordinate (0-100)")
    plt.ylabel("Y Coordinate (0-100)")
    
    # Add coordinate labels
    for x in range(0, 101, 20):
        for y in range(0, 101, 20):
            plt.text(x, y, f"({x},{y})", fontsize=8, alpha=0.6, ha='center', va='center')
    
    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()
    
    return True

def save_points():
    """Save selected points to file"""
    if len(selected_points) == 2:
        print(f"\nSaving points:")
        print(f"START: {selected_points[0]}")
        print(f"GOAL: {selected_points[1]}")
        
        with open('selected_points.txt', 'w') as f:
            f.write(f"{selected_points[0][0]},{selected_points[0][1]}\n")
            f.write(f"{selected_points[1][0]},{selected_points[1][1]}\n")
        
        print("Points saved to selected_points.txt")
        
        # Show conversion to pathfinding coordinates
        start_path = (selected_points[0][0] / 10.0, selected_points[0][1] / 10.0)
        goal_path = (selected_points[1][0] / 10.0, selected_points[1][1] / 10.0)
        
        print(f"\nPathfinding coordinates:")
        print(f"START: {start_path}")
        print(f"GOAL: {goal_path}")
        
        distance = ((goal_path[0] - start_path[0])**2 + (goal_path[1] - start_path[1])**2)**0.5
        print(f"Distance: {distance:.2f} units")
        
        return True
    else:
        print("Error: Need exactly 2 points (start and goal)")
        return False

if __name__ == "__main__":
    if setup_point_selection():
        save_points()
    
    print("\nTo use these points in training:")
    print("1. Run: python train_standalone.py")
    print("2. Choose 'y' when asked about custom points")
    print("3. The training will use your selected start/goal points")
