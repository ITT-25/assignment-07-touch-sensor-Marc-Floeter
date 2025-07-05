import pyglet
import math
import sys
import time
import pandas as pd
from DIPPID import SensorUDP

PORT = 5700
sensor = SensorUDP(PORT)

NUM_TARGETS = int(sys.argv[1])
TARGET_WIDTH = int(sys.argv[2])
TARGET_DISTANCE = int(sys.argv[3])
PARTICIPANT_ID = int(sys.argv[4])

WIDTH = 800
HEIGHT = 800
POINTER_RADIUS = 7
COLOR_POINTER = (0, 255, 0)
COLOR_POINTER_CLICK = (0, 0, 255)
COLOR_TARGET_ACTIVE = (255, 0, 0)
COLOR_TARGET_PASSIVE = (100, 100, 100)


window = pyglet.window.Window(WIDTH, HEIGHT, caption="Fitts Law Sample")
logs = []

class FittsLaw:
    def __init__(self, num_targets, target_w, target_d):
        self.num_targets = num_targets
        self.target_w = target_w
        self.target_d = target_d
        self.active_target = 0
        self.targets = self.setup_targets()
        self.pointer = pyglet.shapes.Circle(0, 0, POINTER_RADIUS, color=COLOR_POINTER)
        self.start_time = None
        

    # sets up targets in a circle and orders them in a way that opposite targets are next to each other
    def setup_targets(self):
        circles = []
        half = self.num_targets // 2
        for i in range(half):
            for j in [i, i + half]:
                angle = 2 * math.pi * j / self.num_targets + math.pi / 2
                x = (WIDTH // 2) + self.target_d * math.cos(angle)
                y = (HEIGHT // 2) + self.target_d * math.sin(angle)
                color = COLOR_TARGET_ACTIVE if j == self.active_target else COLOR_TARGET_PASSIVE
                circle = pyglet.shapes.Circle(x, y, self.target_w, color=color)
                circles.append(circle)
        return circles
    

    # draws the pointer and draws targets as long as the experiment is still running
    def draw(self):
        if self.active_target < self.num_targets:
            for target in self.targets:
                target.draw()
        self.pointer.draw()
        

    def on_move(self, x, y):
        self.pointer.x = x
        self.pointer.y = y


    # colors the pointer while tapping
    def on_click(self, clicked):
        self.pointer.color = COLOR_POINTER_CLICK if clicked == 1 else COLOR_POINTER
        if clicked == 1 and self.hit_test(self.targets[self.active_target], self.pointer.x, self.pointer.y):
            self.log()
            self.update()


    def hit_test(self, obj, x, y):
        dx = x - obj.x   
        dy = y - obj.y
        return dx*dx + dy*dy <= obj.radius * obj.radius


    # appends log data to logs-list. delta time is the time elapsed since the last successful target hit
    # time measurement starts with hitting the first target (-> delta time for first target is 0)
    def log(self):
        current_time = time.time()
        delta_time = (current_time - self.start_time) if self.active_target != 0 else 0
        logs.append([PARTICIPANT_ID, NUM_TARGETS, TARGET_WIDTH, TARGET_DISTANCE, self.active_target, delta_time])
        self.start_time = current_time

    
    # updates the gui and saves log data when experiment done
    def update(self):
        self.active_target += 1
        for i, target in enumerate(self.targets):  
            target.color = COLOR_TARGET_ACTIVE if i == self.active_target else COLOR_TARGET_PASSIVE
        if self.active_target >= self.num_targets:
            logs_df = pd.DataFrame(logs, columns=['pid', 'num_targets', 'target_w', 'target_d', 'iteration', 'time'])
            logs_df.to_csv(f"logs_{time.time()}.csv")             

   

@window.event
def on_draw():
    window.clear()
    fitts.draw()

@window.event
def on_close():
    sensor.disconnect()


# receive and forward events from custom touch sensor to fitts law application 

def handle_movement(data):
    fitts.on_move(data['x'], data['y'])

def handle_tap(is_tapping):
    print(is_tapping)
    fitts.on_click(is_tapping)

sensor.register_callback('movement', handle_movement)
sensor.register_callback('tap', handle_tap)


fitts = FittsLaw(NUM_TARGETS, TARGET_WIDTH, TARGET_DISTANCE)
pyglet.app.run()