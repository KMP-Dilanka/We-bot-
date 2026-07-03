# ==================================================================================
# RoboRoarZ 2025 - Elimination Round | E-Puck Maze Navigator (Webots Controller)
# ----------------------------------------------------------------------------------
# Robot  : E-Puck (standard — encoder motors used for exact navigation)
# Maze   : 12x12 grid, each cell = 25cm x 25cm (total 3m x 3m)
# Sensors: 8 proximity sensors (ps0–ps7), camera, wheel encoders
# ==================================================================================

from controller import Robot, Motor, DistanceSensor, Camera, LED, Gyro
import math
from collections import deque

# ----------------------------------------------------------------------------------
# CONSTANTS
# ----------------------------------------------------------------------------------

CELL_SIZE    = 0.25     
WHEEL_RADIUS = 0.0205   
AXLE_BIAS    = 0.0057 
AXLE_LENGTH  = 0.052 + AXLE_BIAS 
MOVE_SPEED   = 5.0      
TURN_SPEED   = 1.0      
GRID_SIZE    = 12       

PROBE_FRACTION = 8.495/25 
PROBE_FRACTION_1 = 11/25
PROBE_STEPS    = (CELL_SIZE * PROBE_FRACTION) / WHEEL_RADIUS  
PROBE_STEPS_1  = (CELL_SIZE * PROBE_FRACTION_1) / WHEEL_RADIUS
PROBE_SPEED    = MOVE_SPEED * 0.5   

TARGET_ROBOT_ANGLE = math.pi / 2
TURN_STEPS_90 = (TARGET_ROBOT_ANGLE * (AXLE_LENGTH / 2.0)) / WHEEL_RADIUS

WALL_THRESHOLD = 180
STEPS_PER_CELL = CELL_SIZE / WHEEL_RADIUS   

NORTH, EAST, SOUTH, WEST = 0, 1, 2, 3

DIR_DELTA = {
    NORTH: (0,  1),
    EAST:  (1,  0),
    SOUTH: (0, -1),
    WEST:  (-1, 0),
}

HEADING_NAME = {NORTH: 'NORTH', EAST: 'EAST', SOUTH: 'SOUTH', WEST: 'WEST'}


# ==================================================================================
# CLASS: MazeMap
# ==================================================================================
class MazeMap:
    def __init__(self):
        self.walls   = [[{} for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]
        self.visited = [[False] * GRID_SIZE for _ in range(GRID_SIZE)]

    def set_wall(self, x, y, direction, has_wall):
        if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
            self.walls[x][y][direction] = has_wall
            dx, dy   = DIR_DELTA[direction]
            nx, ny   = x + dx, y + dy
            opposite = (direction + 2) % 4
            if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                self.walls[nx][ny][opposite] = has_wall

    def has_wall(self, x, y, direction):
        if not (0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE):
            return True
        w = self.walls[x][y].get(direction, None)
        if w is None:
            dx, dy = DIR_DELTA[direction]
            nx, ny = x + dx, y + dy
            if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                return True
            return False
        return bool(w)

    def mark_visited(self, x, y):
        self.visited[x][y] = True

    def is_visited(self, x, y):
        return self.visited[x][y]


# ==================================================================================
# FUNCTION: bfs_path
# ==================================================================================
def bfs_path(maze_map, start, goal):
    sx, sy = start
    gx, gy = goal
    queue  = deque([(sx, sy, [])])
    seen   = {(sx, sy)}

    while queue:
        cx, cy, path = queue.popleft()
        if (cx, cy) == (gx, gy):
            return path + [(cx, cy)]
        for d in [NORTH, EAST, SOUTH, WEST]:
            if not maze_map.has_wall(cx, cy, d):
                dx, dy = DIR_DELTA[d]
                nx, ny = cx + dx, cy + dy
                if not (0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE):
                    continue
                if (nx, ny) not in seen:
                    seen.add((nx, ny))
                    queue.append((nx, ny, path + [(cx, cy)]))
    return None


# ==================================================================================
# CLASS: MazeRobot
# ==================================================================================
class MazeRobot:
    def __init__(self):
        self.robot = Robot()
        self.ts    = int(self.robot.getBasicTimeStep())
        self.dt    = self.ts / 1000.0

        self.left_motor  = self.robot.getDevice('left wheel motor')
        self.right_motor = self.robot.getDevice('right wheel motor')
        self.left_motor.setPosition(float('inf'))
        self.right_motor.setPosition(float('inf'))
        self.left_motor.setVelocity(0)
        self.right_motor.setVelocity(0)

        self.left_enc  = self.robot.getDevice('left wheel sensor')
        self.right_enc = self.robot.getDevice('right wheel sensor')
        self.left_enc.enable(self.ts)
        self.right_enc.enable(self.ts)

        self.gyro = self.robot.getDevice('gyro')
        self.gyro.enable(self.ts)

        self.ps = []
        for i in range(8):
            s = self.robot.getDevice(f'ps{i}')
            s.enable(self.ts)
            self.ps.append(s)

        self.camera = self.robot.getDevice('camera')
        self.camera.enable(self.ts)

        self.leds = []
        for i in range(8):
            try:
                self.leds.append(self.robot.getDevice(f'led{i}'))
            except Exception:
                pass
        try:
            self.body_led = self.robot.getDevice('body led')
        except Exception:
            self.body_led = None

        self.x       = 11
        self.y       = 0
        self.heading = NORTH

        self.maze_map    = MazeMap()
        self.checkpoints = []
        self.cells_since_recenter = 0


    def step(self):
        return self.robot.step(self.ts) != -1

    def set_speeds(self, left, right):
        self.left_motor.setVelocity(left)
        self.right_motor.setVelocity(right)

    def stop(self):
        self.set_speeds(0, 0)
        self.step()

    def get_encoders(self):
        return self.left_enc.getValue(), self.right_enc.getValue()
    
    def move_forward_one_cell(self):
        l0, r0 = self.get_encoders()
        self.set_speeds(MOVE_SPEED, MOVE_SPEED)
        wall_hit = False
        travelled = 0
        green_checked = False

        while self.step():
            lv, rv    = self.get_encoders()
            travelled = ((lv - l0) + (rv - r0)) / 2.0

            if travelled >= STEPS_PER_CELL:
                break

            if travelled < (STEPS_PER_CELL * 0.30):
                continue

            front_0 = self.ps[0].getValue()
            front_7 = self.ps[7].getValue()
            
            if front_0 > 160 and front_7 > 160 and not green_checked:
                green_checked = True 
                if self._check_green_condition():
                    self.stop()
                    self._record_checkpoint()
                    wall_hit = True
                    break

            front = (front_0 + front_7)/2
            if front > WALL_THRESHOLD * 1.8:
                self.stop()
                wall_hit = True
                break

        self.stop()

        if wall_hit:
            self.maze_map.set_wall(self.x, self.y, self.heading, True)
            l0_rev, r0_rev = self.get_encoders()
            self.set_speeds(-MOVE_SPEED * 0.6, -MOVE_SPEED * 0.6)
            while self.step():
                lv, rv    = self.get_encoders()
                reversed_dist = abs(((lv - l0_rev) + (rv - r0_rev)) / 2.0)
                
                if reversed_dist >= travelled:
                    break
                    
            self.stop()
        
        return not wall_hit

    def rotate(self, direction_change):
        turns_needed = abs(direction_change)
        
        for turn_num in range(turns_needed):
            remaining = TURN_STEPS_90
            start_left_encoder = self.left_enc.getValue()

            if remaining >= 0.5*TURN_STEPS_90:
                K=1
            else:           
                K = remaining/(TURN_STEPS_90*5)

            if direction_change > 0:
               self.set_speeds(K*TURN_SPEED, -K*TURN_SPEED)
            else:
               self.set_speeds(-K*TURN_SPEED, K*TURN_SPEED)

            while self.step():
                current_left_pos = self.left_enc.getValue()
                rotated = abs(current_left_pos - start_left_encoder)
                remaining = TURN_STEPS_90 - rotated

                if remaining <= 0:
                    self.stop()
                    break

            for _ in range(3):
                self.step()

            if turn_num < turns_needed - 1:
               for _ in range(5):
                   self.step()

        self.heading = (self.heading + direction_change) % 4
        return direction_change

    def turn_to(self, target_heading):
        diff = (target_heading - self.heading) % 4
        if diff == 0:
            return
        if diff == 1:            
            n=self.rotate(1)         
        elif diff == 3:
            n=self.rotate(-1)          
        elif diff == 2:
            n=self.rotate(2)
        return n

    def _check_green_condition(self, threshold=0.15):
        img = self.camera.getImage()
        if img is None:
            return False

        width  = self.camera.getWidth()
        height = self.camera.getHeight()

        green_count  = 0
        total_sample = 0
        
        row_start = int(height * 0.5)
        
        for row in range(row_start, height):
            for col in range(0, width, 3):
                r = self.camera.imageGetRed(img, width, col, row)
                g = self.camera.imageGetGreen(img, width, col, row)
                b = self.camera.imageGetBlue(img, width, col, row)
                total_sample += 1
                
                if g > 80 and g > (r * 1.3) and g > (b * 1.3):
                    green_count += 1

        ratio = green_count / max(total_sample, 1)
        return ratio > threshold

    def _record_checkpoint(self):
        pos = (self.x, self.y)
        if pos not in self.checkpoints:
            self.checkpoints.append(pos)
            self.flash_led(times=3)

    def align_to_wall_pid(self):
        Kp = 0.0010  
        Kd = 0.0008  
        TOLERANCE = 1.0 
        MAX_STEPS = 150 
        
        start_l, start_r = self.get_encoders()
        steps = 0
        prev_error = 0
        error = 0  
        
        while self.step() and steps < MAX_STEPS:
            steps += 1
            val0 = self.ps[0].getValue()
            val7 = self.ps[7].getValue()
            
            if val0 < 80 and val7 < 80:
                self.set_speeds(0, 0)
                break
                
            error = val0 - val7

            if abs(error) <= TOLERANCE:
                self.set_speeds(0, 0)
                break 

            derivative = error - prev_error
            turn_speed = (error * Kp) + (derivative * Kd)
            turn_speed = max(-0.4, min(turn_speed, 0.4)) 
            
            if 0 < turn_speed < 0.05: turn_speed = 0.05
            if 0 > turn_speed > -0.05: turn_speed = -0.05

            self.set_speeds(turn_speed, -turn_speed)
            prev_error = error
            
        self.set_speeds(0, 0)

    def recenter_in_maze(self):
        if (self.x, self.y) in self.checkpoints:
            return

        has_y = self.maze_map.has_wall(self.x, self.y, NORTH) or self.maze_map.has_wall(self.x, self.y, SOUTH)
        has_x = self.maze_map.has_wall(self.x, self.y, EAST) or self.maze_map.has_wall(self.x, self.y, WEST)

        if not (has_y and has_x):
            return

        original_heading = self.heading
        y_wall = NORTH if self.maze_map.has_wall(self.x, self.y, NORTH) else SOUTH
        x_wall = EAST if self.maze_map.has_wall(self.x, self.y, EAST) else WEST

        for target_wall in [y_wall, x_wall]:
            self.turn_to(target_wall)
            l0, r0 = self.get_encoders()
            self.set_speeds(PROBE_SPEED, PROBE_SPEED)
            
            green_checked = False
            hit_green = False
            
            while self.step():
                lv, rv = self.get_encoders()
                travelled = ((lv - l0) + (rv - r0)) / 2.0
                
                if travelled > (STEPS_PER_CELL * 0.6):
                    break

                front_0 = self.ps[0].getValue()
                front_7 = self.ps[7].getValue()
                
                if front_0 > 160 and front_7 > 160 and not green_checked:
                    green_checked = True 
                    if self._check_green_condition():
                        self.stop()
                        self._record_checkpoint()
                        hit_green = True
                        break

                front = max(front_0, front_7)
                if front > WALL_THRESHOLD * 1.8:
                    break
            self.stop()
            
            if not hit_green:
                self.align_to_wall_pid()
            
            l0_rev, r0_rev = self.get_encoders()
            self.set_speeds(-PROBE_SPEED, -PROBE_SPEED)
            while self.step():
                lv, rv = self.get_encoders()
                reversed_dist = abs(((lv - l0_rev) + (rv - r0_rev)) / 2.0)
                if reversed_dist >= PROBE_STEPS:
                    break
            self.stop()

            if hit_green:
                break

        self.turn_to(original_heading)
        self.cells_since_recenter = 0  

    def center_in_start_cell(self, initial_heading):
        centered_x = False
        centered_y = False
        walls_used = 0

        for i in range(4):
            if centered_x and centered_y:
                break

            check_heading = (initial_heading + i) % 4
            self.turn_to(check_heading)

            l0, r0 = self.get_encoders()
            self.set_speeds(PROBE_SPEED, PROBE_SPEED)
            hit_wall = False
            travelled = 0

            while self.step():
                lv, rv = self.get_encoders()
                travelled = ((lv - l0) + (rv - r0)) / 2.0
                
                if travelled > (STEPS_PER_CELL * 0.6):
                    break

                front = max(self.ps[0].getValue(), self.ps[7].getValue())
                if front > WALL_THRESHOLD * 1.8:
                    hit_wall = True
                    break

            self.stop()

            if hit_wall:
                self.align_to_wall_pid()
                l0_rev, r0_rev = self.get_encoders()
                self.set_speeds(-PROBE_SPEED, -PROBE_SPEED)
                while self.step():
                    lv, rv = self.get_encoders()
                    reversed_dist = abs(((lv - l0_rev) + (rv - r0_rev)) / 2.0)
                    if reversed_dist >= PROBE_STEPS :
                        break
                self.stop()

                if check_heading in [NORTH, SOUTH]:
                    centered_y = True
                elif check_heading in [EAST, WEST]:
                    centered_x = True
                
                walls_used += 1
            else:
                l0_rev, r0_rev = self.get_encoders()
                self.set_speeds(-PROBE_SPEED, -PROBE_SPEED)
                while self.step():
                    lv, rv = self.get_encoders()
                    reversed_dist = abs(((lv - l0_rev) + (rv - r0_rev)) / 2.0)
                    if reversed_dist >= travelled:
                        break
                self.stop()       

        self.turn_to(initial_heading)

    def probe_for_wall(self, absolute_direction, is_boundary=False):
        original_heading = self.heading
        self.turn_to(absolute_direction)

        for _ in range(5):
            self.step()

        l0, r0    = self.get_encoders()
        emergency = False
        green_checked = False
        self.set_speeds(PROBE_SPEED, PROBE_SPEED)

        while self.step():
            lv, rv    = self.get_encoders()
            travelled = ((lv - l0) + (rv - r0)) / 2.0
            
            if travelled >= PROBE_STEPS_1:                
                break

            front_0 = self.ps[0].getValue()
            front_7 = self.ps[7].getValue()
            
            if max(front_0 , front_7 )>160 and not green_checked:
                green_checked = True 
                if self._check_green_condition():
                    self.stop()
                    self._record_checkpoint()
                    self.set_speeds(PROBE_SPEED, PROBE_SPEED)

            front_max = max(front_0, front_7)
            if front_max > WALL_THRESHOLD :
                emergency = True
                break
                
        self.stop()

        front_val = max(self.ps[0].getValue(), self.ps[7].getValue())
        has_wall  = (front_val > WALL_THRESHOLD) or emergency or is_boundary
        
        if is_boundary and not green_checked:
            if self._check_green_condition():
                self._record_checkpoint()
        
        if has_wall: 
            self.stop()
            for _ in range(5):
                self.step()
            
            if not is_boundary:
                self.align_to_wall_pid()            

        l0, r0 = self.get_encoders()
        self.set_speeds(-PROBE_SPEED, -PROBE_SPEED)
        while self.step():
            lv, rv    = self.get_encoders()
            travelled_reverse = abs(((lv - l0) + (rv - r0)) / 2.0)
            
            if travelled_reverse >= travelled:
                break
                
        self.stop()
        self.turn_to(original_heading)
        return has_wall
    
    def sense_all_walls_probe(self, is_start_cell=False):
        if self.maze_map.is_visited(self.x, self.y):
            return
        
        probe_order = [
            (self.heading + 1) % 4,
            self.heading,
            (self.heading - 1) % 4,
        ]

        if is_start_cell:
            probe_order.append((self.heading + 2) % 4)

        for direction in probe_order[::-1]:
            dx, dy = DIR_DELTA[direction]
            nx, ny = self.x + dx, self.y + dy
            is_boundary = (nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE)
            
            has_wall = self.probe_for_wall(direction, is_boundary=is_boundary)
            self.maze_map.set_wall(self.x, self.y, direction, has_wall)

        self.maze_map.mark_visited(self.x, self.y)

    def flash_led(self, times=3):
        for _ in range(times):
            for led in self.leds: led.set(1)
            if self.body_led: self.body_led.set(1)
            for _ in range(10): self.step()
            for led in self.leds: led.set(0)
            if self.body_led: self.body_led.set(0)
            for _ in range(10): self.step()

    def light_led_continuous(self):
        for led in self.leds: led.set(1)
        if self.body_led: self.body_led.set(1)

    def _wall_follow_step(self):
        vals  = [s.getValue() for s in self.ps]
        sense_front = (vals[0] > WALL_THRESHOLD) or (vals[7] > WALL_THRESHOLD)
        sense_right = (vals[1] > WALL_THRESHOLD) or (vals[2] > WALL_THRESHOLD)
        sense_left  = (vals[5] > WALL_THRESHOLD) or (vals[6] > WALL_THRESHOLD)

        front = sense_front or self.maze_map.has_wall(self.x, self.y, self.heading)
        right = sense_right or self.maze_map.has_wall(self.x, self.y, (self.heading + 1) % 4)
        left  = sense_left  or self.maze_map.has_wall(self.x, self.y, (self.heading - 1) % 4)

        if not left:
            self.rotate(-1)
        elif not front:
            pass
        elif not right:
            self.rotate(1)
        else:
            self.rotate(2)

        ok = self.move_forward_one_cell()
        if ok:
            dx, dy = DIR_DELTA[self.heading]
            self.x = max(0, min(GRID_SIZE - 1, self.x + dx))
            self.y = max(0, min(GRID_SIZE - 1, self.y + dy))

    # ============================================================
    # MAZE EXPLORATION & STRICT SEQUENTIAL TRACKER
    # ============================================================



    # ============================================================
    # MAZE EXPLORATION (OPTIMIZED DEPTH-FIRST SEARCH)
    # ============================================================

    def explore_maze(self):
        """
        Explores the maze using Depth-First Search with Virtual Backtracking.
        Minimizes physical movement by finding the next valid intersection in memory
        before physically driving to it.
        """
        stack = [(self.x, self.y)]

        while stack:
            cx, cy = self.x, self.y
            
            # 1. Sense walls around the current cell and mark it as visited
            self.sense_all_walls_probe()
            
            # 2. Look for any accessible adjacent cells that haven't been visited yet
            unvisited = []
            for d in [NORTH, EAST, SOUTH, WEST]:
                if not self.maze_map.has_wall(cx, cy, d):
                    nx, ny = cx + DIR_DELTA[d][0], cy + DIR_DELTA[d][1]
                    if not self.maze_map.is_visited(nx, ny):
                        unvisited.append((nx, ny, d))
            
            if unvisited:
                # OPTIMIZATION 1: Sort by rotational cost. 
                # min((target - current) % 4, (current - target) % 4) calculates minimum 90-deg turns needed.
                # Straight = 0, Left/Right = 1, U-Turn = 2.
                unvisited.sort(key=lambda n: min((n[2] - self.heading) % 4, (self.heading - n[2]) % 4))
                
                next_x, next_y, needed_heading = unvisited[0]
                
                # Push the current cell to the stack
                stack.append((cx, cy))
                
                # Physically move to the optimal unvisited neighbor
                self.turn_to(needed_heading)
                ok = self.move_forward_one_cell()
                
                if ok:
                    self.x, self.y = next_x, next_y
                    self.cells_since_recenter += 1
                    if self.cells_since_recenter >= 10:
                        self.recenter_in_maze()
            else:
                # OPTIMIZATION 2: Virtual Backtracking
                # Look down the stack in memory to find the closest node that STILL has unvisited paths.
                # Do NOT physically move yet.
                target_cell = None
                while stack:
                    candidate = stack[-1] # Peek at the top of the stack
                    has_options = False
                    
                    for d in [NORTH, EAST, SOUTH, WEST]:
                        if not self.maze_map.has_wall(candidate[0], candidate[1], d):
                            nx, ny = candidate[0] + DIR_DELTA[d][0], candidate[1] + DIR_DELTA[d][1]
                            if not self.maze_map.is_visited(nx, ny):
                                has_options = True
                                break
                                
                    if has_options:
                        target_cell = candidate
                        break
                    else:
                        # This intersection is fully exhausted. Pop it from memory.
                        stack.pop() 
                
                # If the stack is empty, the maze is fully explored!
                if not target_cell:
                    break 
                
                # OPTIMIZATION 3: Direct Driving
                # Now that we know exactly which intersection to go back to, use BFS 
                # to drive there as fast as possible. If we discovered loops, BFS might 
                # even find a faster shortcut back than the way we came!
                while (self.x, self.y) != target_cell:
                    path = bfs_path(self.maze_map, (self.x, self.y), target_cell)
                    
                    if not path or len(path) < 2:
                        break  
                        
                    nx, ny = path[1]
                    dx, dy = nx - self.x, ny - self.y
                    
                    if   dy > 0: needed = NORTH
                    elif dy < 0: needed = SOUTH
                    elif dx > 0: needed = EAST
                    else:        needed = WEST
                    
                    self.turn_to(needed)
                    ok = self.move_forward_one_cell()
                    
                    if ok:
                        self.x, self.y = nx, ny
                        self.sense_all_walls_probe()
                        self.cells_since_recenter += 1
                        if self.cells_since_recenter >= 10:
                            self.recenter_in_maze()
                    else:
                        # If a new wall blocks our shortcut, the map updates and 
                        # BFS calculates a new route on the next while-loop iteration.
                        self.sense_all_walls_probe()

    def compute_final_cell(self):
        if not self.checkpoints:
            return (0, 0)
        sum_x = sum(c[0] for c in self.checkpoints)
        sum_y = sum(c[1] for c in self.checkpoints)
        fx    = sum_x % GRID_SIZE
        fy    = sum_y % GRID_SIZE
        return (fx, fy)

    def goto_final_cell(self, goal):
        for attempt in range(200):
            if (self.x, self.y) == goal:
                return True

            path = bfs_path(self.maze_map, (self.x, self.y), goal)

            if path and len(path) > 1:
                nx, ny = path[1]
                dx, dy = nx - self.x, ny - self.y
                if   dy > 0: needed = NORTH
                elif dy < 0: needed = SOUTH
                elif dx > 0: needed = EAST
                else:        needed = WEST

                self.turn_to(needed)
                ok = self.move_forward_one_cell()
                
                if ok:
                    ddx, ddy = DIR_DELTA[self.heading]
                    self.x = max(0, min(GRID_SIZE - 1, self.x + ddx))
                    self.y = max(0, min(GRID_SIZE - 1, self.y + ddy))
                    self.sense_all_walls_probe()

                    self.cells_since_recenter += 1
                    if self.cells_since_recenter >= 5:
                        self.recenter_in_maze()
            else:
                self._wall_follow_step()
                self.sense_all_walls_probe()

                self.cells_since_recenter += 1
                if self.cells_since_recenter >= 5:
                    self.recenter_in_maze()

        return (self.x, self.y) == goal

    def run(self):
        for _ in range(10):
            self.step()

        self.center_in_start_cell(initial_heading=self.heading)
        self.sense_all_walls_probe(is_start_cell=True)
        self.explore_maze()
        
        final_cell = self.compute_final_cell()
        reached = self.goto_final_cell(final_cell)

        if reached:
           self.light_led_continuous()
        else:
           self.flash_led(times=10)

        # =================================================================
        # FINAL REPORT 
        # =================================================================
        print("\n" + "=" * 60)
        print("  FINAL REPORT: GREEN TILE COORDINATES")
        print("=" * 60)
        
        if not self.checkpoints:
            print("  No green tiles were detected during the run.")
        else:
            for i, (cx, cy) in enumerate(self.checkpoints):
                print(f"  Green Tile {i+1}: X = {cx}, Y = {cy}")
                
        print("-" * 60)
        print(f"  Calculated Target Cell: {final_cell}")
        print("=" * 60 + "\n")

        while self.step():
              pass

# ==================================================================================
# ENTRY POINT
# ==================================================================================
if __name__ == '__main__':
    controller = MazeRobot()
    controller.run()