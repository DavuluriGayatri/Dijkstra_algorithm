import heapq
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Constants
CLEARANCE = 5
ROBOT_RADIUS = 0
MAP_HEIGHT = 500
MAP_WIDTH = 1200

# Actions defined as below
actions = [(1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1), (1, -1), (-1, -1)]

def apply_action(node, action):
    return tuple(np.add(node, action))

# Map creation using OpenCV
def create_map():
    canvas = np.zeros((MAP_HEIGHT, MAP_WIDTH, 3), dtype=np.uint8)
    cv2.rectangle(canvas, (100 + CLEARANCE, 100 + CLEARANCE), (175 - CLEARANCE, 500 - CLEARANCE), (255, 255, 255), thickness=cv2.FILLED)
    cv2.rectangle(canvas, (275 + CLEARANCE, 0 + CLEARANCE), (350 - CLEARANCE, 400 - CLEARANCE), (255, 255, 255), thickness=cv2.FILLED)
    return canvas

def is_valid(node):
    x, y = node
    if x < 0 or x >= MAP_WIDTH or y < 0 or y >= MAP_HEIGHT:
        return False
    return np.all(map_canvas[y, x] != [255, 255, 255])

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

def generate_graph(start, goal):
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
            if new_node not in closed_list and is_valid(new_node):
                new_cost = cost_to_come[current_node] + cost(current_node, new_node)
                if new_node not in cost_to_come or new_cost < cost_to_come[new_node]:
                    cost_to_come[new_node] = new_cost
                    heapq.heappush(open_list, PriorityQueueNode(new_cost, new_node))
                    parents[new_node] = current_node

    return None, None

map_canvas = create_map()

start_x = int(input("Enter the x-coordinate for the start node: "))
start_y = int(input("Enter the y-coordinate for the start node: "))
start_node = (start_x, start_y)

goal_x = int(input("Enter the x-coordinate for the goal node: "))
goal_y = int(input("Enter the y-coordinate for the goal node: "))
goal_node = (goal_x, goal_y)

# Generate Graph and Optimal Path
optimal_path, exploration = generate_graph(start_node, goal_node)

# Visualization
fig, ax = plt.subplots(figsize=(10, 5))
ax.imshow(cv2.cvtColor(map_canvas, cv2.COLOR_BGR2RGB))  
exploration_step_size = 10000  

def animate(i):
    explored_nodes = exploration[:i*exploration_step_size]
    if explored_nodes:
        ax.scatter([node[0] for node in explored_nodes], [node[1] for node in explored_nodes], color='yellow', s=1)
    if i*exploration_step_size >= len(exploration) and optimal_path:
        ax.scatter([node[0] for node in optimal_path], [node[1] for node in optimal_path], color='red', s=1)

total_frames = len(exploration) // exploration_step_size 

ani = FuncAnimation(fig, animate, frames=total_frames, interval=0, repeat=False)


plt.show()
    