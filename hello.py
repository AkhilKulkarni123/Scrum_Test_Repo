#!/usr/bin/env python3
"""
Interactive ASCII Particle Animation System
A terminal-based animation engine with multiple scenes, particle effects, and user controls.

Controls:
- Arrow keys: Move cursor/camera
- Space: Spawn particles at cursor
- 1-5: Switch scenes
- +/-: Adjust particle spawn rate
- P: Pause/unpause
- R: Reset scene
- Q: Quit

Scenes:
1. Particle fountain with gravity
2. Fireworks display
3. Matrix-style rain
4. Spiral galaxy simulation
5. Conway's Game of Life with particles
"""

import sys
import os
import time
import random
import math
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple
import termios
import tty
import select

# ============================================================================
# UTILITY CLASSES AND FUNCTIONS
# ============================================================================

@dataclass
class Vec2:
    """2D vector for position and velocity"""
    x: float
    y: float
    
    def __add__(self, other):
        return Vec2(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vec2(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vec2(self.x * scalar, self.y * scalar)
    
    def length(self):
        return math.sqrt(self.x**2 + self.y**2)
    
    def normalize(self):
        l = self.length()
        if l > 0:
            return Vec2(self.x / l, self.y / l)
        return Vec2(0, 0)

@dataclass
class Particle:
    """Individual particle with physics"""
    pos: Vec2
    vel: Vec2
    age: int = 0
    lifetime: int = 50
    char: str = '*'
    color: int = 37  # ANSI color code
    
    def update(self, gravity: Vec2 = Vec2(0, 0.1), drag: float = 0.99):
        self.vel = self.vel + gravity
        self.vel = self.vel * drag
        self.pos = self.pos + self.vel
        self.age += 1
    
    def is_alive(self):
        return self.age < self.lifetime

class Screen:
    """Terminal screen buffer"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.buffer = [[' ' for _ in range(width)] for _ in range(height)]
        self.color_buffer = [[37 for _ in range(width)] for _ in range(height)]
    
    def clear(self):
        self.buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        self.color_buffer = [[37 for _ in range(self.width)] for _ in range(self.height)]
    
    def set_pixel(self, x: int, y: int, char: str, color: int = 37):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.buffer[y][x] = char
            self.color_buffer[y][x] = color
    
    def draw_text(self, x: int, y: int, text: str, color: int = 37):
        for i, char in enumerate(text):
            self.set_pixel(x + i, y, char, color)
    
    def render(self):
        # Move cursor to top-left
        sys.stdout.write('\033[H')
        
        for y in range(self.height):
            line_parts = []
            current_color = None
            
            for x in range(self.width):
                color = self.color_buffer[y][x]
                char = self.buffer[y][x]
                
                if color != current_color:
                    if current_color is not None:
                        line_parts.append('\033[0m')  # Reset
                    line_parts.append(f'\033[{color}m')
                    current_color = color
                
                line_parts.append(char)
            
            if current_color is not None:
                line_parts.append('\033[0m')
            
            sys.stdout.write(''.join(line_parts) + '\n')
        
        sys.stdout.flush()

class InputHandler:
    """Non-blocking keyboard input handler"""
    def __init__(self):
        self.old_settings = None
    
    def __enter__(self):
        self.old_settings = termios.tcgetattr(sys.stdin)
        tty.setcbreak(sys.stdin.fileno())
        return self
    
    def __exit__(self, type, value, traceback):
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self.old_settings)
    
    def get_key(self, timeout=0.001):
        if select.select([sys.stdin], [], [], timeout)[0]:
            key = sys.stdin.read(1)
            # Handle arrow keys
            if key == '\x1b':
                next1 = sys.stdin.read(1)
                next2 = sys.stdin.read(1)
                if next1 == '[':
                    return {'A': 'UP', 'B': 'DOWN', 'C': 'RIGHT', 'D': 'LEFT'}.get(next2, None)
            return key
        return None

# ============================================================================
# SCENE IMPLEMENTATIONS
# ============================================================================

class ParticleFountain:
    """Scene 1: Particle fountain with gravity"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.cursor = Vec2(width // 2, height // 2)
        self.spawn_rate = 5
        self.frame = 0
    
    def update(self):
        # Spawn new particles from cursor
        if self.frame % (max(1, 10 - self.spawn_rate)) == 0:
            for _ in range(random.randint(1, 3)):
                angle = random.uniform(-math.pi/3, -2*math.pi/3)
                speed = random.uniform(1, 2.5)
                vel = Vec2(math.cos(angle) * speed, math.sin(angle) * speed)
                
                colors = [31, 33, 36, 35, 34]  # Red, yellow, cyan, magenta, blue
                self.particles.append(Particle(
                    pos=Vec2(self.cursor.x, self.cursor.y),
                    vel=vel,
                    lifetime=random.randint(30, 60),
                    char=random.choice(['*', '•', '○', '·']),
                    color=random.choice(colors)
                ))
        
        # Update particles
        for p in self.particles:
            p.update(gravity=Vec2(0, 0.15), drag=0.98)
        
        # Remove dead particles
        self.particles = [p for p in self.particles if p.is_alive()]
        self.frame += 1
    
    def render(self, screen: Screen):
        screen.clear()
        
        # Draw particles
        for p in self.particles:
            alpha = 1.0 - (p.age / p.lifetime)
            if alpha > 0.7:
                char = p.char
            elif alpha > 0.4:
                char = '·'
            else:
                char = '.'
            
            screen.set_pixel(int(p.pos.x), int(p.pos.y), char, p.color)
        
        # Draw cursor
        screen.set_pixel(int(self.cursor.x), int(self.cursor.y), '+', 32)
        
        # Draw info
        screen.draw_text(2, 1, "PARTICLE FOUNTAIN", 36)
        screen.draw_text(2, 2, f"Particles: {len(self.particles)}", 37)
        screen.draw_text(2, 3, f"Spawn rate: {self.spawn_rate}", 37)

class Fireworks:
    """Scene 2: Fireworks display"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.rockets: List[Particle] = []
        self.frame = 0
        self.spawn_rate = 5
    
    def spawn_rocket(self):
        x = random.randint(10, self.width - 10)
        rocket = Particle(
            pos=Vec2(x, self.height - 2),
            vel=Vec2(random.uniform(-0.2, 0.2), random.uniform(-3, -4)),
            lifetime=random.randint(20, 35),
            char='|',
            color=33
        )
        self.rockets.append(rocket)
    
    def explode(self, pos: Vec2):
        num_particles = random.randint(20, 40)
        colors = [31, 33, 32, 36, 35, 34]
        color = random.choice(colors)
        
        for _ in range(num_particles):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(0.5, 2.5)
            vel = Vec2(math.cos(angle) * speed, math.sin(angle) * speed)
            
            self.particles.append(Particle(
                pos=Vec2(pos.x, pos.y),
                vel=vel,
                lifetime=random.randint(15, 35),
                char=random.choice(['*', '•', '○', '+']),
                color=color
            ))
    
    def update(self):
        # Spawn rockets
        if self.frame % (max(1, 30 - self.spawn_rate * 3)) == 0:
            self.spawn_rocket()
        
        # Update rockets
        for rocket in self.rockets:
            rocket.update(gravity=Vec2(0, 0.1), drag=0.97)
        
        # Explode rockets that reached peak
        for rocket in self.rockets[:]:
            if rocket.age > rocket.lifetime * 0.6 or not rocket.is_alive():
                self.explode(rocket.pos)
                self.rockets.remove(rocket)
        
        # Update particles
        for p in self.particles:
            p.update(gravity=Vec2(0, 0.08), drag=0.96)
        
        self.particles = [p for p in self.particles if p.is_alive()]
        self.frame += 1
    
    def render(self, screen: Screen):
        screen.clear()
        
        # Draw rockets
        for rocket in self.rockets:
            screen.set_pixel(int(rocket.pos.x), int(rocket.pos.y), rocket.char, rocket.color)
            # Trail
            trail_y = int(rocket.pos.y) + 1
            if 0 <= trail_y < self.height:
                screen.set_pixel(int(rocket.pos.x), trail_y, '·', 33)
        
        # Draw particles
        for p in self.particles:
            screen.set_pixel(int(p.pos.x), int(p.pos.y), p.char, p.color)
        
        screen.draw_text(2, 1, "FIREWORKS DISPLAY", 36)
        screen.draw_text(2, 2, f"Active: {len(self.particles) + len(self.rockets)}", 37)

class MatrixRain:
    """Scene 3: Matrix-style digital rain"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.columns = []
        self.spawn_rate = 5
        
        for x in range(0, width, 2):
            self.columns.append({
                'x': x,
                'y': random.randint(-20, 0),
                'speed': random.uniform(0.3, 1.2),
                'chars': [],
                'active': random.random() > 0.5
            })
    
    def update(self):
        chars = "アイウエオカキクケコサシスセソタチツテトナニヌネノハヒフヘホマミムメモヤユヨラリルレロワヲン0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        
        for col in self.columns:
            if col['active']:
                col['y'] += col['speed']
                
                # Add new character at top
                if random.random() > 0.7:
                    col['chars'].append({
                        'char': random.choice(chars),
                        'y': col['y'],
                        'age': 0
                    })
                
                # Update existing characters
                for char in col['chars']:
                    char['age'] += 1
                
                # Remove old characters
                col['chars'] = [c for c in col['chars'] if c['age'] < 20]
                
                # Reset column
                if col['y'] > self.height + 10:
                    col['y'] = random.randint(-20, -5)
                    col['speed'] = random.uniform(0.3, 1.2)
                    col['chars'] = []
                    col['active'] = random.random() > 0.3
            else:
                if random.random() > 0.99:
                    col['active'] = True
    
    def render(self, screen: Screen):
        screen.clear()
        
        for col in self.columns:
            for char_data in col['chars']:
                y = int(char_data['y'] - col['y'] + char_data['age'])
                if 0 <= y < self.height:
                    # Fade effect
                    if char_data['age'] < 2:
                        color = 37  # White (bright)
                    elif char_data['age'] < 5:
                        color = 32  # Green (bright)
                    elif char_data['age'] < 10:
                        color = 32  # Green
                    else:
                        color = 32  # Green (dim)
                    
                    screen.set_pixel(col['x'], y, char_data['char'], color)
        
        screen.draw_text(2, 1, "MATRIX RAIN", 32)

class SpiralGalaxy:
    """Scene 4: Spiral galaxy simulation"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.center = Vec2(width / 2, height / 2)
        self.particles: List[Particle] = []
        self.spawn_rate = 5
        self.angle_offset = 0
        
        # Create initial galaxy
        for _ in range(200):
            self.spawn_star()
    
    def spawn_star(self):
        # Spiral arm generation
        arm = random.randint(0, 3)
        distance = random.uniform(5, min(self.width, self.height) / 2.5)
        angle = random.uniform(0, 2 * math.pi) + arm * (math.pi / 2)
        
        # Add spiral curve
        angle += distance * 0.1
        
        pos = Vec2(
            self.center.x + math.cos(angle) * distance,
            self.center.y + math.sin(angle) * distance * 0.5  # Flatten for ASCII
        )
        
        # Orbital velocity
        orbital_speed = 0.15 / (distance / 10 + 1)
        vel = Vec2(
            -math.sin(angle) * orbital_speed,
            math.cos(angle) * orbital_speed * 0.5
        )
        
        colors = [37, 36, 34, 35]
        self.particles.append(Particle(
            pos=pos,
            vel=vel,
            lifetime=1000,
            char=random.choice(['·', '•', '*', '.']),
            color=random.choice(colors)
        ))
    
    def update(self):
        self.angle_offset += 0.01
        
        # Spawn new stars occasionally
        if random.random() > 0.95 and len(self.particles) < 300:
            self.spawn_star()
        
        # Update particles with gravity toward center
        for p in self.particles:
            to_center = self.center - p.pos
            distance = to_center.length()
            
            if distance > 1:
                # Gravitational force
                force = to_center.normalize() * (0.5 / distance)
                p.vel = p.vel + force
                p.vel = p.vel * 0.99  # Slight drag
                p.pos = p.pos + p.vel
            
            p.age += 1
        
        # Remove particles that drifted too far
        self.particles = [p for p in self.particles 
                         if (p.pos - self.center).length() < min(self.width, self.height)]
    
    def render(self, screen: Screen):
        screen.clear()
        
        # Draw particles
        for p in self.particles:
            distance = (p.pos - self.center).length()
            # Size based on distance (perspective)
            if distance < 10:
                char = '*'
            elif distance < 20:
                char = '•'
            else:
                char = '·'
            
            screen.set_pixel(int(p.pos.x), int(p.pos.y), char, p.color)
        
        # Draw center
        screen.set_pixel(int(self.center.x), int(self.center.y), '◉', 33)
        
        screen.draw_text(2, 1, "SPIRAL GALAXY", 36)
        screen.draw_text(2, 2, f"Stars: {len(self.particles)}", 37)

class GameOfLife:
    """Scene 5: Conway's Game of Life with particle effects"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.grid = [[False for _ in range(width)] for _ in range(height)]
        self.particles: List[Particle] = []
        self.frame = 0
        self.spawn_rate = 5
        
        # Initialize with random pattern
        self.randomize()
    
    def randomize(self):
        for y in range(self.height):
            for x in range(self.width):
                self.grid[y][x] = random.random() > 0.7
    
    def count_neighbors(self, x: int, y: int) -> int:
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx, ny = x + dx, y + dy
                if 0 <= nx < self.width and 0 <= ny < self.height:
                    if self.grid[ny][nx]:
                        count += 1
        return count
    
    def update(self):
        # Update Game of Life every N frames
        if self.frame % 5 == 0:
            new_grid = [[False for _ in range(self.width)] for _ in range(self.height)]
            
            for y in range(self.height):
                for x in range(self.width):
                    neighbors = self.count_neighbors(x, y)
                    
                    if self.grid[y][x]:
                        # Cell is alive
                        new_grid[y][x] = neighbors in [2, 3]
                        if not new_grid[y][x]:
                            # Cell died - create particle effect
                            if random.random() > 0.8:
                                vel = Vec2(random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5))
                                self.particles.append(Particle(
                                    pos=Vec2(x, y),
                                    vel=vel,
                                    lifetime=random.randint(10, 20),
                                    char='·',
                                    color=31
                                ))
                    else:
                        # Cell is dead
                        if neighbors == 3:
                            new_grid[y][x] = True
                            # Cell born - create particle effect
                            if random.random() > 0.8:
                                vel = Vec2(random.uniform(-0.3, 0.3), random.uniform(-0.3, 0.3))
                                self.particles.append(Particle(
                                    pos=Vec2(x, y),
                                    vel=vel,
                                    lifetime=random.randint(10, 20),
                                    char='*',
                                    color=32
                                ))
            
            self.grid = new_grid
        
        # Update particles
        for p in self.particles:
            p.update(drag=0.95)
        
        self.particles = [p for p in self.particles if p.is_alive()]
        self.frame += 1
    
    def render(self, screen: Screen):
        screen.clear()
        
        # Draw cells
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x]:
                    screen.set_pixel(x, y, '█', 32)
        
        # Draw particles
        for p in self.particles:
            screen.set_pixel(int(p.pos.x), int(p.pos.y), p.char, p.color)
        
        screen.draw_text(2, 1, "GAME OF LIFE", 36)
        screen.draw_text(2, 2, f"Generation: {self.frame // 5}", 37)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

class AnimationEngine:
    """Main animation engine"""
    def __init__(self):
        # Get terminal size
        size = os.get_terminal_size()
        self.width = size.columns
        self.height = size.lines - 1
        
        self.screen = Screen(self.width, self.height)
        self.scenes = [
            ParticleFountain(self.width, self.height),
            Fireworks(self.width, self.height),
            MatrixRain(self.width, self.height),
            SpiralGalaxy(self.width, self.height),
            GameOfLife(self.width, self.height)
        ]
        self.current_scene = 0
        self.paused = False
        self.running = True
    
    def handle_input(self, key):
        scene = self.scenes[self.current_scene]
        
        if key == 'q' or key == 'Q':
            self.running = False
        elif key == 'p' or key == 'P':
            self.paused = not self.paused
        elif key == 'r' or key == 'R':
            # Reset current scene
            self.scenes[self.current_scene] = type(scene)(self.width, self.height)
        elif key in ['1', '2', '3', '4', '5']:
            self.current_scene = int(key) - 1
        elif key == '+' or key == '=':
            if hasattr(scene, 'spawn_rate'):
                scene.spawn_rate = min(10, scene.spawn_rate + 1)
        elif key == '-' or key == '_':
            if hasattr(scene, 'spawn_rate'):
                scene.spawn_rate = max(1, scene.spawn_rate - 1)
        elif hasattr(scene, 'cursor'):
            # Handle cursor movement for scenes that support it
            if key == 'UP' and scene.cursor.y > 0:
                scene.cursor.y -= 1
            elif key == 'DOWN' and scene.cursor.y < self.height - 1:
                scene.cursor.y += 1
            elif key == 'LEFT' and scene.cursor.x > 0:
                scene.cursor.x -= 1
            elif key == 'RIGHT' and scene.cursor.x < self.width - 1:
                scene.cursor.x += 1
    
    def run(self):
        # Clear screen and hide cursor
        sys.stdout.write('\033[2J\033[?25l')
        sys.stdout.flush()
        
        try:
            with InputHandler() as input_handler:
                last_time = time.time()
                
                while self.running:
                    current_time = time.time()
                    dt = current_time - last_time
                    
                    # Handle input
                    key = input_handler.get_key()
                    if key:
                        self.handle_input(key)
                    
                    # Update and render
                    if not self.paused:
                        self.scenes[self.current_scene].update()
                    
                    self.scenes[self.current_scene].render(self.screen)
                    
                    # Draw UI
                    self.draw_ui()
                    
                    self.screen.render()
                    
                    # Target 30 FPS
                    frame_time = time.time() - current_time
                    sleep_time = max(0, (1.0 / 30.0) - frame_time)
                    time.sleep(sleep_time)
                    
                    last_time = current_time
        
        finally:
            # Show cursor and clear screen
            sys.stdout.write('\033[?25h\033[2J\033[H')
            sys.stdout.flush()
    
    def draw_ui(self):
        # Bottom bar
        y = self.height - 1
        controls = "1-5:Scenes | ARROWS:Move | SPACE:Spawn | +/-:Rate | P:Pause | R:Reset | Q:Quit"
        self.screen.draw_text(2, y, controls, 33)
        
        # Status
        status = f"Scene {self.current_scene + 1}/5"
        if self.paused:
            status += " [PAUSED]"
        self.screen.draw_text(self.width - len(status) - 2, y, status, 33)

def main():
    if sys.platform == 'win32':
        print("This program requires a Unix-like terminal (Linux/Mac/WSL)")
        return
    
    try:
        engine = AnimationEngine()
        engine.run()
    except KeyboardInterrupt:
        sys.stdout.write('\033[?25h\033[2J\033[H')
        sys.stdout.flush()
        print("\nAnimation stopped.")

if __name__ == '__main__':
    main()