import heapq
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
MAP_HEIGHT = 500
MAP_WIDTH = 1200
CLEARANCE = 5
ROBOT_RADIUS = 0

# Actions defined as below
actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

def apply_action(node, action):
    return tuple(np.add(node, action))

def create_map_canvas():
    canvas = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    # Obstacle 1 (rectangle)
    cv2.rectangle(canvas, (100, 100), (175, 500), (0, 150, 255), thickness=cv2.FILLED)
    # Obstacle 2 (rectangle)
    cv2.rectangle(canvas, (275, 0), (350, 400), (150, 255, 255), thickness=cv2.FILLED)
    # Obstacle 3 (Hexagon)
    hexagon_pts = np.array([[650, 120], [537, 185], [537, 315], [650, 380], [763, 315], [763, 185]], dtype=np.int32)
    cv2.fillPoly(canvas, [hexagon_pts], color=(0, 255, 0))
    # Obstacle 4 (C-shaped)
    c_shape_rect_pts = np.array([[900, 450], [1100, 450], [1100, 50], [900, 50], [900, 125], [1020, 125], [1020, 375], [900, 375]], dtype=np.int32)
    cv2.fillPoly(canvas, [c_shape_rect_pts], color=(200, 0, 255))

    return canvas

def is_valid(node, map_canvas):
    x, y = node
    if x < CLEARANCE or x >= MAP_WIDTH - CLEARANCE or y < CLEARANCE or y >= MAP_HEIGHT - CLEARANCE:
        return False

    # Check for obstacles with clearance
    clearance_area = map_canvas[max(0, y - CLEARANCE):min(MAP_HEIGHT, y + CLEARANCE + 1),
                                 max(0, x - CLEARANCE):min(MAP_WIDTH, x + CLEARANCE + 1)]
    return np.all(clearance_area != [255, 255, 255])

class PriorityQueueNode:
    def __init__(self, cost, node):
        self.cost = cost
        self.node = node

    def __lt__(self, other):
        return self.cost < other.cost

def backtrack(parents, start, goal):
    path = [goal]
    current_node = goal
    while current_node != start:
        current_node = parents[current_node]
        path.insert(0, current_node)
    return path

def cost(node1, node2):
    return np.sqrt((node1[0] - node2[0])**2 + (node1[1] - node2[1])**2)

def generate_graph(start, goal, map_canvas):
    open_list = []
    heapq.heappush(open_list, PriorityQueueNode(0, start))
    closed_list = set()
    exploration = []
    parents = {}
    cost_to_come = {start: 0}

    while open_list:
        current_node = heapq.heappop(open_list).node
        exploration.append(current_node)

        if current_node == goal:
            return backtrack(parents, start, goal), exploration

        closed_list.add(current_node)

        for action in actions:
            new_node = apply_action(current_node, action)
            if new_node not in closed_list and is_valid(new_node, map_canvas):
                new_cost = cost_to_come[current_node] + cost(current_node, new_node)
                if new_node not in cost_to_come or new_cost < cost_to_come[new_node]:
                    cost_to_come[new_node] = new_cost
                    heapq.heappush(open_list, PriorityQueueNode(new_cost, new_node))
                    parents[new_node] = current_node

    return None, exploration

def get_user_input(map_canvas):
    while True:
        start_x = int(input("Enter the x-coordinate for the start node: "))
        start_y = int(input("Enter the y-coordinate for the start node: "))
        goal_x = int(input("Enter the x-coordinate for the goal node: "))
        goal_y = int(input("Enter the y-coordinate for the goal node: "))

        start_node = (start_x, start_y)
        goal_node = (goal_x, goal_y)

        if not is_valid(start_node, map_canvas) or not is_valid(goal_node, map_canvas):
            print("Invalid input! Please make sure the start and goal nodes are not in the obstacle space. Try again.")
        else:
            return start_node, goal_node

if __name__ == "__main__":
    map_canvas = create_map_canvas()
    start_node, goal_node = get_user_input(map_canvas)

    # Generate Graph and Optimal Path
    optimal_path, exploration = generate_graph(start_node, goal_node, map_canvas)

    # Visualization
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(cv2.cvtColor(map_canvas, cv2.COLOR_BGR2RGB), origin='lower')
    exploration_step_size = 10000

    def animate(i):
        explored_nodes = exploration[:i * exploration_step_size]
        if explored_nodes:
            ax.scatter([node[0] for node in explored_nodes], [node[1] for node in explored_nodes], color='yellow', s=1)
        if i * exploration_step_size >= len(exploration) and optimal_path:
            ax.scatter([node[0] for node in optimal_path], [node[1] for node in optimal_path], color='red', s=1)
        
        # Visualize start node in green
        ax.scatter(start_node[0], start_node[1], color='green', marker='o', s=50, label='Start')
        # Visualize goal node in blue
        ax.scatter(goal_node[0], goal_node[1], color='blue', marker='o', s=50, label='Goal')
    
    total_frames = len(exploration) // exploration_step_size + (2 if optimal_path else 1) # Total frames calculation

    ani = FuncAnimation(fig, animate, frames=total_frames, interval=0, repeat=False)

    # Print Optimal Path
    print("Optimal Path:", optimal_path)

    plt.show()
