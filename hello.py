import pygame
import math
import random
from pygame import gfxdraw

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 1400, 900
FPS = 60
AU = 149.6e6 * 1000  # Astronomical Unit in meters
G = 6.67428e-11  # Gravitational constant
SCALE = 200 / AU  # 1 AU = 200 pixels
TIMESTEP = 3600 * 24  # 1 day in seconds

# Colors
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
YELLOW = (255, 255, 0)
GREY = (128, 128, 128)
ORANGE = (255, 140, 0)
BLUE = (100, 149, 237)
RED = (188, 39, 50)
DARK_GREY = (80, 78, 81)
LIGHT_BLUE = (173, 216, 230)
PALE_YELLOW = (245, 222, 179)
PALE_BLUE = (176, 224, 230)

class CelestialBody:
    def __init__(self, x, y, radius, color, mass, name):
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.mass = mass
        self.name = name
        self.orbit = []
        self.sun = False
        self.distance_to_sun = 0
        self.x_vel = 0
        self.y_vel = 0
        
    def draw(self, win, offset_x, offset_y, zoom):
        x = self.x * SCALE * zoom + WIDTH / 2 + offset_x
        y = self.y * SCALE * zoom + HEIGHT / 2 + offset_y
        
        # Draw orbit trail
        if len(self.orbit) > 2:
            updated_points = []
            for point in self.orbit:
                px, py = point
                px = px * SCALE * zoom + WIDTH / 2 + offset_x
                py = py * SCALE * zoom + HEIGHT / 2 + offset_y
                updated_points.append((px, py))
            
            if len(updated_points) > 1:
                pygame.draw.lines(win, self.color, False, updated_points, 1)
        
        # Draw planet with glow effect
        radius = max(int(self.radius * zoom), 2)
        
        # Glow effect
        for i in range(3):
            glow_radius = radius + (3 - i) * 2
            glow_color = tuple(min(255, c + 40) for c in self.color)
            gfxdraw.filled_circle(win, int(x), int(y), glow_radius, (*glow_color, 50))
        
        # Main body
        pygame.draw.circle(win, self.color, (int(x), int(y)), radius)
        
        # Add some texture for larger bodies
        if radius > 5:
            pygame.draw.circle(win, tuple(max(0, c - 30) for c in self.color), 
                             (int(x - radius/3), int(y - radius/3)), max(1, radius // 4))
        
        # Draw name
        if zoom > 0.5:
            font = pygame.font.SysFont("comicsans", max(12, int(14 * zoom)))
            text = font.render(self.name, 1, WHITE)
            win.blit(text, (x - text.get_width() / 2, y - text.get_height() / 2 - radius - 15))
    
    def attraction(self, other):
        other_x, other_y = other.x, other.y
        distance_x = other_x - self.x
        distance_y = other_y - self.y
        distance = math.sqrt(distance_x ** 2 + distance_y ** 2)
        
        if other.sun:
            self.distance_to_sun = distance
        
        force = G * self.mass * other.mass / distance ** 2
        theta = math.atan2(distance_y, distance_x)
        force_x = math.cos(theta) * force
        force_y = math.sin(theta) * force
        return force_x, force_y
    
    def update_position(self, bodies):
        total_fx = total_fy = 0
        for body in bodies:
            if self == body:
                continue
            
            fx, fy = self.attraction(body)
            total_fx += fx
            total_fy += fy
        
        self.x_vel += total_fx / self.mass * TIMESTEP
        self.y_vel += total_fy / self.mass * TIMESTEP
        
        self.x += self.x_vel * TIMESTEP
        self.y += self.y_vel * TIMESTEP
        self.orbit.append((self.x, self.y))
        
        if len(self.orbit) > 1000:
            self.orbit.pop(0)

class Star:
    def __init__(self, x, y, brightness):
        self.x = x
        self.y = y
        self.brightness = brightness
        self.twinkle_offset = random.uniform(0, math.pi * 2)

def create_starfield(num_stars):
    stars = []
    for _ in range(num_stars):
        x = random.randint(0, WIDTH)
        y = random.randint(0, HEIGHT)
        brightness = random.randint(100, 255)
        stars.append(Star(x, y, brightness))
    return stars

def draw_starfield(win, stars, frame_count):
    for star in stars:
        twinkle = abs(math.sin(frame_count * 0.02 + star.twinkle_offset))
        brightness = int(star.brightness * (0.5 + 0.5 * twinkle))
        color = (brightness, brightness, brightness)
        pygame.draw.circle(win, color, (star.x, star.y), 1)

def draw_ui(win, bodies, paused, speed_multiplier, zoom, show_info):
    font = pygame.font.SysFont("comicsans", 16)
    
    # Control instructions
    instructions = [
        "Controls:",
        "SPACE - Pause/Resume",
        "Arrow Keys - Pan view",
        "+/- - Zoom in/out",
        "UP/DOWN - Speed up/slow down time",
        "I - Toggle planet info",
        "R - Reset view"
    ]
    
    y_offset = 10
    for instruction in instructions:
        text = font.render(instruction, 1, WHITE)
        win.blit(text, (10, y_offset))
        y_offset += 20
    
    # Status
    status_text = f"{'PAUSED' if paused else 'RUNNING'} | Speed: {speed_multiplier}x | Zoom: {zoom:.1f}x"
    text = font.render(status_text, 1, YELLOW)
    win.blit(text, (WIDTH - text.get_width() - 10, 10))
    
    # Planet info
    if show_info:
        info_y = HEIGHT - 200
        info_bg = pygame.Surface((300, 180))
        info_bg.set_alpha(200)
        info_bg.fill((20, 20, 40))
        win.blit(info_bg, (WIDTH - 310, info_y))
        
        title = font.render("Planet Distances (AU):", 1, YELLOW)
        win.blit(title, (WIDTH - 300, info_y + 10))
        
        y = info_y + 35
        for body in bodies:
            if not body.sun and body.distance_to_sun > 0:
                dist_au = body.distance_to_sun / AU
                text = font.render(f"{body.name}: {dist_au:.2f} AU", 1, body.color)
                win.blit(text, (WIDTH - 290, y))
                y += 20

def main():
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Solar System Simulator")
    clock = pygame.time.Clock()
    
    # Create starfield
    stars = create_starfield(200)
    
    # Create celestial bodies
    sun = CelestialBody(0, 0, 30, YELLOW, 1.98892 * 10**30, "Sun")
    sun.sun = True
    
    mercury = CelestialBody(0.387 * AU, 0, 4, DARK_GREY, 3.30 * 10**23, "Mercury")
    mercury.y_vel = -47.4 * 1000
    
    venus = CelestialBody(0.723 * AU, 0, 7, PALE_YELLOW, 4.8685 * 10**24, "Venus")
    venus.y_vel = -35.02 * 1000
    
    earth = CelestialBody(-1 * AU, 0, 8, BLUE, 5.9742 * 10**24, "Earth")
    earth.y_vel = 29.783 * 1000
    
    mars = CelestialBody(-1.524 * AU, 0, 6, RED, 6.39 * 10**23, "Mars")
    mars.y_vel = 24.077 * 1000
    
    jupiter = CelestialBody(5.204 * AU, 0, 18, ORANGE, 1.898 * 10**27, "Jupiter")
    jupiter.y_vel = -13.06 * 1000
    
    saturn = CelestialBody(9.583 * AU, 0, 16, PALE_YELLOW, 5.683 * 10**26, "Saturn")
    saturn.y_vel = -9.68 * 1000
    
    uranus = CelestialBody(-19.191 * AU, 0, 12, LIGHT_BLUE, 8.681 * 10**25, "Uranus")
    uranus.y_vel = 6.80 * 1000
    
    neptune = CelestialBody(-30.07 * AU, 0, 12, PALE_BLUE, 1.024 * 10**26, "Neptune")
    neptune.y_vel = 5.43 * 1000
    
    bodies = [sun, mercury, venus, earth, mars, jupiter, saturn, uranus, neptune]
    
    # Camera controls
    offset_x = 0
    offset_y = 0
    zoom = 1.0
    pan_speed = 10
    
    # Simulation controls
    paused = False
    speed_multiplier = 1
    show_info = True
    frame_count = 0
    
    run = True
    while run:
        clock.tick(FPS)
        frame_count += 1
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                    zoom = min(3.0, zoom + 0.1)
                elif event.key == pygame.K_MINUS:
                    zoom = max(0.3, zoom - 0.1)
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(10, speed_multiplier + 1)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(1, speed_multiplier - 1)
                elif event.key == pygame.K_i:
                    show_info = not show_info
                elif event.key == pygame.K_r:
                    offset_x = 0
                    offset_y = 0
                    zoom = 1.0
        
        # Handle continuous key presses for panning
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            offset_x += pan_speed
        if keys[pygame.K_RIGHT]:
            offset_x -= pan_speed
        # Removed vertical panning to avoid conflict with speed controls
        
        win.fill(BLACK)
        
        # Draw starfield
        draw_starfield(win, stars, frame_count)
        
        # Update and draw bodies
        for body in bodies:
            if not paused:
                for _ in range(speed_multiplier):
                    body.update_position(bodies)
            body.draw(win, offset_x, offset_y, zoom)
        
        # Draw UI
        draw_ui(win, bodies, paused, speed_multiplier, zoom, show_info)
        
        pygame.display.update()
    
    pygame.quit()

if __name__ == "__main__":
    main()