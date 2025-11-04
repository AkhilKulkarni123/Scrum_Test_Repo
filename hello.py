"""
Interactive Particle-Based Fluid Simulation with SPH (Smoothed Particle Hydrodynamics)
Features: Real-time physics, color gradients, interactive spawning, gravity control
Requirements: pip install pygame numpy
"""
import sys
import time
import os
import pygame
import pygame.gfxdraw
import pygame.freetype
import pygame.surfarray
import pygame.pixelarray
import pygame.time
import pygame.locals

import numpy as np
import ctypes
import math
import random
from dataclasses import dataclass
from typing import List, Tuple

# Constants
WIDTH, HEIGHT = 1200, 800
FPS = 60
PARTICLE_RADIUS = 4
MAX_PARTICLES = 1500
SPAWN_RATE = 15

# Physics constants
SMOOTHING_RADIUS = 30.0
TARGET_DENSITY = 0.5
PRESSURE_MULTIPLIER = 50.0
GRAVITY = 500.0
DAMPING = 0.98
VISCOSITY = 0.1
COLLISION_DAMPING = 0.5

# Color scheme
BACKGROUND = (10, 10, 25)
PARTICLE_COLORS = [
    (100, 150, 255),
    (120, 180, 255),
    (80, 200, 255),
    (60, 220, 255),
]

@dataclass
class Particle:
    """Represents a single fluid particle with position, velocity, and properties"""
    x: float
    y: float
    vx: float = 0.0
    vy: float = 0.0
    density: float = 0.0
    pressure: float = 0.0
    color_idx: int = 0
    
    def update_position(self, dt: float):
        """Update particle position based on velocity"""
        self.x += self.vx * dt
        self.y += self.vy * dt
    
    def apply_force(self, fx: float, fy: float, dt: float):
        """Apply force to particle (F = ma, assuming mass = 1)"""
        self.vx += fx * dt
        self.vy += fy * dt

class SpatialHash:
    """Spatial hash grid for efficient neighbor finding"""
    def __init__(self, cell_size: float, width: int, height: int):
        self.cell_size = cell_size
        self.cols = int(width / cell_size) + 1
        self.rows = int(height / cell_size) + 1
        self.grid = {}
    
    def clear(self):
        """Clear the spatial hash grid"""
        self.grid.clear()
    
    def hash_pos(self, x: float, y: float) -> Tuple[int, int]:
        """Convert world position to grid cell"""
        col = int(x / self.cell_size)
        row = int(y / self.cell_size)
        return (col, row)
    
    def insert(self, particle: Particle, index: int):
        """Insert particle into spatial hash"""
        cell = self.hash_pos(particle.x, particle.y)
        if cell not in self.grid:
            self.grid[cell] = []
        self.grid[cell].append(index)
    
    def get_nearby(self, x: float, y: float) -> List[int]:
        """Get all particle indices near a position"""
        nearby = []
        cell = self.hash_pos(x, y)
        
        # Check 3x3 grid of cells
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                check_cell = (cell[0] + dx, cell[1] + dy)
                if check_cell in self.grid:
                    nearby.extend(self.grid[check_cell])
        
        return nearby

class FluidSimulation:
    """Main fluid simulation using SPH method"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.particles: List[Particle] = []
        self.spatial_hash = SpatialHash(SMOOTHING_RADIUS, width, height)
        self.gravity_strength = GRAVITY
        self.paused = False
        self.spawn_mode = True
        
    def add_particle(self, x: float, y: float):
        """Add a new particle at position"""
        if len(self.particles) < MAX_PARTICLES:
            # Add some randomness to initial velocity
            vx = random.uniform(-50, 50)
            vy = random.uniform(-50, 50)
            color_idx = random.randint(0, len(PARTICLE_COLORS) - 1)
            particle = Particle(x, y, vx, vy, color_idx=color_idx)
            self.particles.append(particle)
    
    def spawn_particles_at(self, x: float, y: float, count: int):
        """Spawn multiple particles in a circle"""
        for _ in range(count):
            angle = random.uniform(0, 2 * math.pi)
            radius = random.uniform(0, 20)
            px = x + math.cos(angle) * radius
            py = y + math.sin(angle) * radius
            self.add_particle(px, py)
    
    def smoothing_kernel(self, distance: float, radius: float) -> float:
        """Poly6 smoothing kernel for SPH"""
        if distance >= radius:
            return 0.0
        
        volume = (math.pi * radius ** 4) / 6.0
        value = max(0, radius ** 2 - distance ** 2)
        return value ** 3 / volume
    
    def smoothing_kernel_derivative(self, distance: float, radius: float) -> float:
        """Gradient of Spiky smoothing kernel"""
        if distance >= radius or distance < 0.0001:
            return 0.0
        
        volume = (math.pi * radius ** 5) / 10.0
        value = radius - distance
        return -3 * value ** 2 / volume
    
    def calculate_densities(self):
        """Calculate density for each particle using SPH"""
        for i, particle in enumerate(self.particles):
            density = 0.0
            nearby = self.spatial_hash.get_nearby(particle.x, particle.y)
            
            for j in nearby:
                if j >= len(self.particles):
                    continue
                    
                other = self.particles[j]
                dx = other.x - particle.x
                dy = other.y - particle.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                influence = self.smoothing_kernel(distance, SMOOTHING_RADIUS)
                density += influence
            
            particle.density = density
            particle.pressure = PRESSURE_MULTIPLIER * (density - TARGET_DENSITY)
    
    def calculate_pressure_forces(self):
        """Calculate pressure and viscosity forces"""
        for i, particle in enumerate(self.particles):
            pressure_fx = 0.0
            pressure_fy = 0.0
            viscosity_fx = 0.0
            viscosity_fy = 0.0
            
            nearby = self.spatial_hash.get_nearby(particle.x, particle.y)
            
            for j in nearby:
                if i == j or j >= len(self.particles):
                    continue
                
                other = self.particles[j]
                dx = other.x - particle.x
                dy = other.y - particle.y
                distance = math.sqrt(dx * dx + dy * dy)
                
                if distance < 0.0001:
                    continue
                
                # Pressure force
                dir_x = dx / distance
                dir_y = dy / distance
                
                slope = self.smoothing_kernel_derivative(distance, SMOOTHING_RADIUS)
                shared_pressure = (particle.pressure + other.pressure) / 2.0
                
                pressure_fx -= shared_pressure * dir_x * slope / other.density if other.density > 0 else 0
                pressure_fy -= shared_pressure * dir_y * slope / other.density if other.density > 0 else 0
                
                # Viscosity force
                influence = self.smoothing_kernel(distance, SMOOTHING_RADIUS)
                viscosity_fx += (other.vx - particle.vx) * influence * VISCOSITY
                viscosity_fy += (other.vy - particle.vy) * influence * VISCOSITY
            
            # Store forces to apply later
            particle.pressure_fx = pressure_fx
            particle.pressure_fy = pressure_fy
            particle.viscosity_fx = viscosity_fx
            particle.viscosity_fy = viscosity_fy
    
    def update(self, dt: float):
        """Update simulation one timestep"""
        if self.paused:
            return
        
        # Rebuild spatial hash
        self.spatial_hash.clear()
        for i, particle in enumerate(self.particles):
            self.spatial_hash.insert(particle, i)
        
        # SPH calculations
        self.calculate_densities()
        self.calculate_pressure_forces()
        
        # Apply forces and update positions
        for particle in self.particles:
            # Apply pressure and viscosity
            particle.apply_force(particle.pressure_fx, particle.pressure_fy, dt)
            particle.apply_force(particle.viscosity_fx, particle.viscosity_fy, dt)
            
            # Apply gravity
            particle.apply_force(0, self.gravity_strength, dt)
            
            # Update position
            particle.update_position(dt)
            
            # Apply damping
            particle.vx *= DAMPING
            particle.vy *= DAMPING
            
            # Boundary collisions
            if particle.x < PARTICLE_RADIUS:
                particle.x = PARTICLE_RADIUS
                particle.vx *= -COLLISION_DAMPING
            elif particle.x > self.width - PARTICLE_RADIUS:
                particle.x = self.width - PARTICLE_RADIUS
                particle.vx *= -COLLISION_DAMPING
            
            if particle.y < PARTICLE_RADIUS:
                particle.y = PARTICLE_RADIUS
                particle.vy *= -COLLISION_DAMPING
            elif particle.y > self.height - PARTICLE_RADIUS:
                particle.y = self.height - PARTICLE_RADIUS
                particle.vy *= -COLLISION_DAMPING
    
    def draw(self, surface: pygame.Surface):
        """Render particles to screen"""
        surface.fill(BACKGROUND)
        
        # Draw particles with density-based coloring
        for particle in self.particles:
            # Color intensity based on density
            density_factor = min(particle.density / (TARGET_DENSITY * 2), 1.0)
            base_color = PARTICLE_COLORS[particle.color_idx]
            
            color = (
                int(base_color[0] * (0.5 + 0.5 * density_factor)),
                int(base_color[1] * (0.5 + 0.5 * density_factor)),
                int(base_color[2] * (0.5 + 0.5 * density_factor))
            )
            
            # Draw particle
            pygame.draw.circle(surface, color, (int(particle.x), int(particle.y)), PARTICLE_RADIUS)
            
            # Draw glow effect for high density
            if density_factor > 0.7:
                glow_radius = PARTICLE_RADIUS + 2
                glow_color = (*color, 50)
                glow_surface = pygame.Surface((glow_radius * 2, glow_radius * 2), pygame.SRCALPHA)
                pygame.draw.circle(glow_surface, glow_color, (glow_radius, glow_radius), glow_radius)
                surface.blit(glow_surface, (int(particle.x) - glow_radius, int(particle.y) - glow_radius))

def draw_ui(surface: pygame.Surface, simulation: FluidSimulation, fps: float):
    """Draw UI overlay with instructions and stats"""
    font = pygame.font.Font(None, 28)
    small_font = pygame.font.Font(None, 22)
    
    # Instructions
    instructions = [
        "LEFT CLICK - Spawn particles",
        "SPACE - Pause/Resume",
        "R - Reset simulation",
        "UP/DOWN - Adjust gravity",
        "C - Clear all particles",
    ]
    
    y_offset = 10
    for instruction in instructions:
        text = small_font.render(instruction, True, (200, 200, 200))
        surface.blit(text, (10, y_offset))
        y_offset += 25
    
    # Stats
    stats = [
        f"Particles: {len(simulation.particles)}/{MAX_PARTICLES}",
        f"FPS: {int(fps)}",
        f"Gravity: {int(simulation.gravity_strength)}",
        f"Status: {'PAUSED' if simulation.paused else 'RUNNING'}",
    ]
    
    y_offset = HEIGHT - 110
    for stat in stats:
        text = small_font.render(stat, True, (150, 255, 150))
        surface.blit(text, (10, y_offset))
        y_offset += 25

def main():
    """Main game loop"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Fluid Simulation - SPH")
    clock = pygame.time.Clock()
    
    simulation = FluidSimulation(WIDTH, HEIGHT)
    
    # Add initial particles
    for _ in range(200):
        x = random.uniform(WIDTH * 0.3, WIDTH * 0.7)
        y = random.uniform(HEIGHT * 0.2, HEIGHT * 0.5)
        simulation.add_particle(x, y)
    
    running = True
    mouse_pressed = False
    spawn_timer = 0
    
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.02)  # Cap dt to prevent instability
        
        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    mouse_pressed = True
            
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    mouse_pressed = False
            
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    simulation.paused = not simulation.paused
                
                elif event.key == pygame.K_r:
                    simulation.particles.clear()
                    for _ in range(200):
                        x = random.uniform(WIDTH * 0.3, WIDTH * 0.7)
                        y = random.uniform(HEIGHT * 0.2, HEIGHT * 0.5)
                        simulation.add_particle(x, y)
                
                elif event.key == pygame.K_c:
                    simulation.particles.clear()
                
                elif event.key == pygame.K_UP:
                    simulation.gravity_strength += 50
                
                elif event.key == pygame.K_DOWN:
                    simulation.gravity_strength = max(0, simulation.gravity_strength - 50)
        
        # Spawn particles on mouse hold
        if mouse_pressed:
            spawn_timer += dt
            if spawn_timer >= 1.0 / 60:  # Spawn every frame
                mouse_x, mouse_y = pygame.mouse.get_pos()
                simulation.spawn_particles_at(mouse_x, mouse_y, SPAWN_RATE)
                spawn_timer = 0
        
        # Update simulation
        simulation.update(dt)
        
        # Render
        simulation.draw(screen)
        draw_ui(screen, simulation, clock.get_fps())
        
        pygame.display.flip()
    
    pygame.quit()

if __name__ == "__main__":
    main()