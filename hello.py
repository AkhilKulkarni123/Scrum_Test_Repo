#!/usr/bin/env python3
"""
ASCII Particle Physics Simulator
A terminal-based particle system with gravity, collisions, and interactive effects.
Press keys 1-7 to spawn different particle effects!
"""

import curses
import random
import math
import time
from dataclasses import dataclass
from typing import List, Tuple
from collections import deque

@dataclass
class Particle:
    x: float
    y: float
    vx: float
    vy: float
    life: float
    max_life: float
    char: str
    color: int
    mass: float = 1.0
    trail: deque = None
    
    def __post_init__(self):
        if self.trail is None:
            self.trail = deque(maxlen=5)

class ParticleSystem:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.gravity = 0.3
        self.wind = 0.0
        self.time = 0
        self.effects = {
            '1': self.spawn_firework,
            '2': self.spawn_fountain,
            '3': self.spawn_explosion,
            '4': self.spawn_rain,
            '5': self.spawn_spiral,
            '6': self.spawn_wave,
            '7': self.spawn_galaxy
        }
        
    def spawn_firework(self, x, y):
        """Colorful firework explosion"""
        colors = [curses.COLOR_RED, curses.COLOR_YELLOW, curses.COLOR_MAGENTA, 
                  curses.COLOR_CYAN, curses.COLOR_GREEN]
        color = random.choice(colors)
        
        for _ in range(80):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 8)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            chars = ['*', '✦', '✧', '◦', '•']
            p = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=random.uniform(1.5, 3.0),
                max_life=3.0,
                char=random.choice(chars),
                color=color,
                mass=random.uniform(0.5, 1.5)
            )
            self.particles.append(p)
    
    def spawn_fountain(self, x, y):
        """Upward fountain of particles"""
        for _ in range(40):
            vx = random.uniform(-2, 2)
            vy = random.uniform(-12, -8)
            color = random.choice([curses.COLOR_CYAN, curses.COLOR_BLUE, curses.COLOR_WHITE])
            
            p = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=random.uniform(2.0, 4.0),
                max_life=4.0,
                char=random.choice(['~', '≈', '∼', '○']),
                color=color,
                mass=random.uniform(0.8, 1.2)
            )
            self.particles.append(p)
    
    def spawn_explosion(self, x, y):
        """Violent explosion with shockwave"""
        # Core explosion
        for _ in range(100):
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(5, 15)
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            p = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=random.uniform(0.5, 2.0),
                max_life=2.0,
                char=random.choice(['#', '@', '%', '&']),
                color=random.choice([curses.COLOR_RED, curses.COLOR_YELLOW]),
                mass=random.uniform(0.3, 0.8)
            )
            self.particles.append(p)
        
        # Shockwave ring
        for i in range(50):
            angle = (i / 50) * 2 * math.pi
            speed = 12
            vx = math.cos(angle) * speed
            vy = math.sin(angle) * speed
            
            p = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=0.8,
                max_life=0.8,
                char='o',
                color=curses.COLOR_WHITE,
                mass=0.1
            )
            self.particles.append(p)
    
    def spawn_rain(self, x, y):
        """Cascading rain effect"""
        for _ in range(60):
            offset_x = random.uniform(-30, 30)
            p = Particle(
                x=x + offset_x, y=y,
                vx=random.uniform(-1, 1),
                vy=random.uniform(8, 12),
                life=random.uniform(3.0, 5.0),
                max_life=5.0,
                char=random.choice(['|', '¦', '│']),
                color=curses.COLOR_BLUE,
                mass=1.0
            )
            self.particles.append(p)
    
    def spawn_spiral(self, x, y):
        """Spiraling particles"""
        num_particles = 60
        for i in range(num_particles):
            angle = (i / num_particles) * 4 * math.pi
            radius = 8
            speed = 5
            
            vx = math.cos(angle) * speed + math.sin(angle) * 2
            vy = math.sin(angle) * speed - math.cos(angle) * 2
            
            colors = [curses.COLOR_MAGENTA, curses.COLOR_CYAN, curses.COLOR_YELLOW]
            color = colors[i % len(colors)]
            
            p = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=random.uniform(2.0, 4.0),
                max_life=4.0,
                char=random.choice(['◉', '◎', '●', '○']),
                color=color,
                mass=0.5
            )
            self.particles.append(p)
    
    def spawn_wave(self, x, y):
        """Sine wave pattern"""
        for i in range(80):
            offset = (i - 40) * 2
            angle = (i / 80) * 2 * math.pi
            
            vx = offset * 0.3
            vy = math.sin(angle) * 8
            
            p = Particle(
                x=x, y=y,
                vx=vx, vy=vy,
                life=random.uniform(2.0, 3.5),
                max_life=3.5,
                char=random.choice(['~', '≈', '∿']),
                color=random.choice([curses.COLOR_CYAN, curses.COLOR_GREEN]),
                mass=0.7
            )
            self.particles.append(p)
    
    def spawn_galaxy(self, x, y):
        """Rotating galaxy effect"""
        num_arms = 5
        particles_per_arm = 20
        
        for arm in range(num_arms):
            base_angle = (arm / num_arms) * 2 * math.pi
            
            for i in range(particles_per_arm):
                t = i / particles_per_arm
                angle = base_angle + t * 4 * math.pi
                radius = t * 15
                
                px = math.cos(angle) * radius
                py = math.sin(angle) * radius
                
                # Orbital velocity
                speed = 3 / (1 + t * 2)
                vx = -math.sin(angle) * speed
                vy = math.cos(angle) * speed
                
                colors = [curses.COLOR_WHITE, curses.COLOR_YELLOW, curses.COLOR_CYAN]
                
                p = Particle(
                    x=x + px, y=y + py,
                    vx=vx, vy=vy,
                    life=random.uniform(3.0, 6.0),
                    max_life=6.0,
                    char=random.choice(['*', '·', '•', '✦']),
                    color=random.choice(colors),
                    mass=0.3
                )
                self.particles.append(p)
    
    def update(self, dt):
        """Update all particles"""
        self.time += dt
        self.wind = math.sin(self.time * 0.5) * 1.5
        
        particles_to_remove = []
        
        for p in self.particles:
            # Store trail position
            p.trail.append((p.x, p.y))
            
            # Apply forces
            p.vy += self.gravity * p.mass * dt * 60
            p.vx += self.wind * dt * 10
            
            # Apply drag
            drag = 0.99
            p.vx *= drag
            p.vy *= drag
            
            # Update position
            p.x += p.vx * dt * 60
            p.y += p.vy * dt * 60
            
            # Update life
            p.life -= dt
            
            # Boundary bouncing
            if p.x < 0:
                p.x = 0
                p.vx *= -0.7
            elif p.x >= self.width:
                p.x = self.width - 1
                p.vx *= -0.7
                
            if p.y >= self.height - 1:
                p.y = self.height - 1
                p.vy *= -0.6
                p.vx *= 0.8
                if abs(p.vy) < 0.5:
                    p.life = min(p.life, 0.5)
            
            # Mark dead particles
            if p.life <= 0:
                particles_to_remove.append(p)
        
        # Remove dead particles
        for p in particles_to_remove:
            self.particles.remove(p)
    
    def render(self, screen):
        """Render particles to screen"""
        # Create buffer
        buffer = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        color_buffer = [[0 for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw particles
        for p in self.particles:
            x, y = int(p.x), int(p.y)
            
            # Draw trail
            if len(p.trail) > 1:
                for i, (tx, ty) in enumerate(list(p.trail)[:-1]):
                    tx, ty = int(tx), int(ty)
                    if 0 <= tx < self.width and 0 <= ty < self.height:
                        alpha = i / len(p.trail)
                        if buffer[ty][tx] == ' ':
                            buffer[ty][tx] = '·' if alpha > 0.5 else '.'
                            color_buffer[ty][tx] = p.color
            
            # Draw particle
            if 0 <= x < self.width and 0 <= y < self.height:
                # Brightness based on life
                life_ratio = p.life / p.max_life
                if life_ratio > 0.7:
                    char = p.char
                elif life_ratio > 0.4:
                    char = '.' if p.char in ['*', '✦', '✧'] else p.char
                else:
                    char = '.'
                
                buffer[y][x] = char
                color_buffer[y][x] = p.color
        
        # Render to screen
        for y in range(min(self.height, curses.LINES - 1)):
            for x in range(min(self.width, curses.COLS)):
                char = buffer[y][x]
                color = color_buffer[y][x]
                
                if char != ' ':
                    try:
                        screen.addstr(y, x, char, curses.color_pair(color + 1))
                    except curses.error:
                        pass

def main(stdscr):
    # Setup
    curses.curs_set(0)  # Hide cursor
    stdscr.nodelay(1)   # Non-blocking input
    stdscr.timeout(0)
    
    # Initialize colors
    curses.start_color()
    curses.use_default_colors()
    
    # Setup color pairs
    colors = [
        curses.COLOR_BLACK,
        curses.COLOR_RED,
        curses.COLOR_GREEN,
        curses.COLOR_YELLOW,
        curses.COLOR_BLUE,
        curses.COLOR_MAGENTA,
        curses.COLOR_CYAN,
        curses.COLOR_WHITE
    ]
    
    for i, color in enumerate(colors):
        curses.init_pair(i + 1, color, -1)
    
    # Get screen dimensions
    height, width = stdscr.getmaxyx()
    
    # Create particle system
    ps = ParticleSystem(width, height - 3)
    
    # Game loop
    last_time = time.time()
    frame_count = 0
    fps = 0
    
    # Welcome message
    instructions = [
        "╔═══════════════════════════════════════════════════════════╗",
        "║  ASCII PARTICLE PHYSICS SIMULATOR                        ║",
        "║  Press 1-7 to spawn different effects:                   ║",
        "║  [1] Firework  [2] Fountain  [3] Explosion  [4] Rain     ║",
        "║  [5] Spiral    [6] Wave      [7] Galaxy     [Q] Quit     ║",
        "╚═══════════════════════════════════════════════════════════╝"
    ]
    
    # Spawn initial effect
    ps.spawn_firework(width // 2, height // 2)
    
    while True:
        current_time = time.time()
        dt = current_time - last_time
        last_time = current_time
        
        # Calculate FPS
        frame_count += 1
        if frame_count % 30 == 0:
            fps = int(1 / dt) if dt > 0 else 0
        
        # Handle input
        try:
            key = stdscr.getkey()
            
            if key.lower() == 'q':
                break
            elif key in ps.effects:
                # Spawn at random position
                x = random.randint(width // 4, 3 * width // 4)
                y = random.randint(height // 4, 3 * height // 4)
                ps.effects[key](x, y)
        except:
            pass
        
        # Auto-spawn effects occasionally
        if random.random() < 0.01:
            effect = random.choice(list(ps.effects.values()))
            x = random.randint(width // 4, 3 * width // 4)
            y = random.randint(height // 4, 3 * height // 4)
            effect(x, y)
        
        # Update
        ps.update(dt)
        
        # Render
        stdscr.clear()
        ps.render(stdscr)
        
        # Draw UI
        try:
            for i, line in enumerate(instructions):
                if i < height - 1:
                    stdscr.addstr(height - len(instructions) + i - 1, 
                                max(0, (width - len(line)) // 2), 
                                line[:width], 
                                curses.color_pair(7))
            
            # Stats
            stats = f" Particles: {len(ps.particles)} | FPS: {fps} | Wind: {ps.wind:.1f} "
            stdscr.addstr(0, 0, stats, curses.color_pair(6))
        except curses.error:
            pass
        
        stdscr.refresh()
        
        # Frame rate limiting
        time.sleep(0.016)  # ~60 FPS

if __name__ == "__main__":
    try:
        curses.wrapper(main)
    except KeyboardInterrupt:
        print("\nParticle simulator closed. Thanks for playing!")
    except Exception as e:
        print(f"\nError: {e}")
        print("Make sure your terminal supports color and is large enough!")