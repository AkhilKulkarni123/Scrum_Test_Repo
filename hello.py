import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import random
from dataclasses import dataclass
from typing import Tuple, List
import colorsys

@dataclass
class TerrainConfig:
    """Configuration for terrain generation"""
    size: int = 200
    octaves: int = 6
    persistence: float = 0.5
    lacunarity: float = 2.0
    scale: float = 100.0
    seed: int = None
    
class PerlinNoise:
    """Perlin noise generator for natural-looking terrain"""
    
    def __init__(self, seed=None):
        self.seed = seed if seed else random.randint(0, 1000000)
        np.random.seed(self.seed)
        self.permutation = np.random.permutation(256)
        self.p = np.concatenate([self.permutation, self.permutation])
        
    def fade(self, t):
        """Smoothstep interpolation"""
        return t * t * t * (t * (t * 6 - 15) + 10)
    
    def lerp(self, t, a, b):
        """Linear interpolation"""
        return a + t * (b - a)
    
    def grad(self, hash_val, x, y):
        """Gradient function"""
        h = hash_val & 3
        u = x if h < 2 else y
        v = y if h < 2 else x
        return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)
    
    def noise(self, x, y):
        """Generate 2D Perlin noise value"""
        xi = int(np.floor(x)) & 255
        yi = int(np.floor(y)) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        
        u = self.fade(xf)
        v = self.fade(yf)
        
        aa = self.p[self.p[xi] + yi]
        ab = self.p[self.p[xi] + yi + 1]
        ba = self.p[self.p[xi + 1] + yi]
        bb = self.p[self.p[xi + 1] + yi + 1]
        
        x1 = self.lerp(u, self.grad(aa, xf, yf), self.grad(ba, xf - 1, yf))
        x2 = self.lerp(u, self.grad(ab, xf, yf - 1), self.grad(bb, xf - 1, yf - 1))
        
        return self.lerp(v, x1, x2)

class TerrainGenerator:
    """Generates procedural terrain using multi-octave Perlin noise"""
    
    def __init__(self, config: TerrainConfig):
        self.config = config
        self.perlin = PerlinNoise(config.seed)
        self.terrain = None
        
    def generate_heightmap(self) -> np.ndarray:
        """Generate terrain heightmap using fractional Brownian motion"""
        heightmap = np.zeros((self.config.size, self.config.size))
        
        for i in range(self.config.size):
            for j in range(self.config.size):
                amplitude = 1.0
                frequency = 1.0
                noise_height = 0.0
                
                for octave in range(self.config.octaves):
                    sample_x = (i / self.config.scale) * frequency
                    sample_y = (j / self.config.scale) * frequency
                    
                    perlin_value = self.perlin.noise(sample_x, sample_y)
                    noise_height += perlin_value * amplitude
                    
                    amplitude *= self.config.persistence
                    frequency *= self.config.lacunarity
                
                heightmap[i, j] = noise_height
        
        # Normalize to 0-1 range
        heightmap = (heightmap - heightmap.min()) / (heightmap.max() - heightmap.min())
        self.terrain = heightmap
        return heightmap
    
    def apply_erosion(self, iterations: int = 3) -> np.ndarray:
        """Simulate basic hydraulic erosion"""
        if self.terrain is None:
            self.generate_heightmap()
        
        eroded = self.terrain.copy()
        
        for _ in range(iterations):
            for i in range(1, self.config.size - 1):
                for j in range(1, self.config.size - 1):
                    # Get neighbor heights
                    neighbors = [
                        eroded[i-1, j], eroded[i+1, j],
                        eroded[i, j-1], eroded[i, j+1]
                    ]
                    
                    # Erode if higher than all neighbors
                    if eroded[i, j] > max(neighbors):
                        diff = eroded[i, j] - max(neighbors)
                        eroded[i, j] -= diff * 0.1
        
        self.terrain = eroded
        return eroded
    
    def add_features(self):
        """Add interesting terrain features like peaks and valleys"""
        if self.terrain is None:
            self.generate_heightmap()
        
        # Add mountain peaks
        num_peaks = random.randint(3, 7)
        for _ in range(num_peaks):
            cx, cy = random.randint(20, self.config.size-20), random.randint(20, self.config.size-20)
            radius = random.randint(15, 30)
            height = random.uniform(0.3, 0.5)
            
            for i in range(max(0, cx-radius), min(self.config.size, cx+radius)):
                for j in range(max(0, cy-radius), min(self.config.size, cy+radius)):
                    dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                    if dist < radius:
                        factor = (1 - dist/radius) ** 2
                        self.terrain[i, j] += height * factor
        
        # Add valleys
        num_valleys = random.randint(2, 4)
        for _ in range(num_valleys):
            cx, cy = random.randint(20, self.config.size-20), random.randint(20, self.config.size-20)
            radius = random.randint(10, 20)
            depth = random.uniform(0.1, 0.2)
            
            for i in range(max(0, cx-radius), min(self.config.size, cx+radius)):
                for j in range(max(0, cy-radius), min(self.config.size, cy+radius)):
                    dist = np.sqrt((i-cx)**2 + (j-cy)**2)
                    if dist < radius:
                        factor = (1 - dist/radius) ** 2
                        self.terrain[i, j] -= depth * factor
        
        # Normalize again
        self.terrain = (self.terrain - self.terrain.min()) / (self.terrain.max() - self.terrain.min())

class TerrainVisualizer:
    """Visualizes terrain with various rendering modes"""
    
    def __init__(self, terrain: np.ndarray):
        self.terrain = terrain
        self.size = terrain.shape[0]
        
    def create_custom_colormap(self) -> List:
        """Create a natural terrain colormap"""
        colors = [
            (0.0, (0.2, 0.3, 0.8)),    # Deep water
            (0.25, (0.3, 0.5, 0.9)),   # Shallow water
            (0.3, (0.8, 0.8, 0.6)),    # Beach
            (0.35, (0.3, 0.6, 0.2)),   # Lowlands
            (0.5, (0.2, 0.5, 0.1)),    # Forest
            (0.65, (0.4, 0.4, 0.3)),   # Hills
            (0.8, (0.5, 0.5, 0.5)),    # Mountains
            (1.0, (1.0, 1.0, 1.0))     # Snow peaks
        ]
        return colors
    
    def get_terrain_color(self, height: float) -> Tuple[float, float, float]:
        """Get color based on height"""
        colors = self.create_custom_colormap()
        
        for i in range(len(colors) - 1):
            h1, c1 = colors[i]
            h2, c2 = colors[i + 1]
            
            if h1 <= height <= h2:
                t = (height - h1) / (h2 - h1)
                r = c1[0] + t * (c2[0] - c1[0])
                g = c1[1] + t * (c2[1] - c1[1])
                b = c1[2] + t * (c2[2] - c1[2])
                return (r, g, b)
        
        return colors[-1][1]
    
    def create_color_array(self) -> np.ndarray:
        """Create full color array for terrain"""
        colors = np.zeros((self.size, self.size, 3))
        for i in range(self.size):
            for j in range(self.size):
                colors[i, j] = self.get_terrain_color(self.terrain[i, j])
        return colors
    
    def plot_3d(self, elev: float = 30, azim: float = 45, show_wireframe: bool = False):
        """Create 3D visualization of terrain"""
        fig = plt.figure(figsize=(15, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(0, self.size, self.size)
        y = np.linspace(0, self.size, self.size)
        X, Y = np.meshgrid(x, y)
        Z = self.terrain * 50  # Scale height for better visualization
        
        colors = self.create_color_array()
        
        if show_wireframe:
            ax.plot_wireframe(X, Y, Z, color='black', alpha=0.3, linewidth=0.5)
        
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, 
                               shade=True, antialiased=True,
                               linewidth=0, alpha=0.9)
        
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Height', fontsize=12)
        ax.set_title('Procedurally Generated Terrain', fontsize=16, fontweight='bold')
        
        ax.view_init(elev=elev, azim=azim)
        ax.set_zlim(0, 50)
        
        # Add lighting effect
        ax.set_facecolor('#87CEEB')
        
        plt.tight_layout()
        return fig, ax
    
    def plot_topographic(self):
        """Create topographic map view"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        
        # Height map with contours
        colors = self.create_color_array()
        ax1.imshow(colors, origin='lower', interpolation='bilinear')
        
        contours = ax1.contour(self.terrain, levels=15, colors='black', 
                               alpha=0.4, linewidths=0.5)
        ax1.clabel(contours, inline=True, fontsize=8, fmt='%.2f')
        ax1.set_title('Topographic Map with Contours', fontsize=14, fontweight='bold')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        
        # 3D contour visualization
        ax2.imshow(self.terrain, cmap='terrain', origin='lower')
        contours2 = ax2.contour(self.terrain, levels=20, colors='white', 
                                alpha=0.6, linewidths=1)
        ax2.set_title('Heightmap with Dense Contours', fontsize=14, fontweight='bold')
        ax2.set_xlabel('X')
        ax2.set_ylabel('Y')
        
        plt.tight_layout()
        return fig
    
    def plot_shaded_relief(self):
        """Create shaded relief map"""
        fig, ax = plt.subplots(figsize=(12, 10))
        
        # Calculate gradients for shading
        dy, dx = np.gradient(self.terrain)
        slope = np.sqrt(dx**2 + dy**2)
        
        # Normalize slope for shading
        slope_norm = (slope - slope.min()) / (slope.max() - slope.min())
        
        # Create shaded terrain
        colors = self.create_color_array()
        shaded = colors * (0.5 + 0.5 * (1 - slope_norm[:, :, np.newaxis]))
        shaded = np.clip(shaded, 0, 1)
        
        ax.imshow(shaded, origin='lower')
        ax.set_title('Shaded Relief Map', fontsize=16, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(False)
        
        plt.tight_layout()
        return fig
    
    def create_rotating_animation(self, filename: str = None):
        """Create rotating 3D animation"""
        fig = plt.figure(figsize=(12, 9))
        ax = fig.add_subplot(111, projection='3d')
        
        x = np.linspace(0, self.size, self.size)
        y = np.linspace(0, self.size, self.size)
        X, Y = np.meshgrid(x, y)
        Z = self.terrain * 50
        
        colors = self.create_color_array()
        
        surf = ax.plot_surface(X, Y, Z, facecolors=colors, 
                               shade=True, antialiased=True,
                               linewidth=0, alpha=0.9)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Height')
        ax.set_title('Rotating Terrain View')
        ax.set_zlim(0, 50)
        ax.set_facecolor('#87CEEB')
        
        def rotate(frame):
            ax.view_init(elev=30, azim=frame)
            return surf,
        
        anim = FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2),
                           interval=50, blit=False)
        
        if filename:
            anim.save(filename, writer='pillow', fps=20)
        
        return fig, anim

class TerrainAnalyzer:
    """Analyzes terrain properties"""
    
    def __init__(self, terrain: np.ndarray):
        self.terrain = terrain
        
    def calculate_statistics(self) -> dict:
        """Calculate terrain statistics"""
        stats = {
            'mean_height': np.mean(self.terrain),
            'max_height': np.max(self.terrain),
            'min_height': np.min(self.terrain),
            'std_dev': np.std(self.terrain),
            'roughness': self.calculate_roughness()
        }
        return stats
    
    def calculate_roughness(self) -> float:
        """Calculate terrain roughness using gradient"""
        dy, dx = np.gradient(self.terrain)
        roughness = np.mean(np.sqrt(dx**2 + dy**2))
        return roughness
    
    def find_peaks(self, threshold: float = 0.8) -> List[Tuple[int, int]]:
        """Find terrain peaks above threshold"""
        peaks = []
        for i in range(1, self.terrain.shape[0] - 1):
            for j in range(1, self.terrain.shape[1] - 1):
                if self.terrain[i, j] > threshold:
                    # Check if it's a local maximum
                    neighbors = [
                        self.terrain[i-1, j], self.terrain[i+1, j],
                        self.terrain[i, j-1], self.terrain[i, j+1],
                        self.terrain[i-1, j-1], self.terrain[i-1, j+1],
                        self.terrain[i+1, j-1], self.terrain[i+1, j+1]
                    ]
                    if self.terrain[i, j] >= max(neighbors):
                        peaks.append((i, j))
        return peaks
    
    def find_valleys(self, threshold: float = 0.2) -> List[Tuple[int, int]]:
        """Find terrain valleys below threshold"""
        valleys = []
        for i in range(1, self.terrain.shape[0] - 1):
            for j in range(1, self.terrain.shape[1] - 1):
                if self.terrain[i, j] < threshold:
                    # Check if it's a local minimum
                    neighbors = [
                        self.terrain[i-1, j], self.terrain[i+1, j],
                        self.terrain[i, j-1], self.terrain[i, j+1],
                        self.terrain[i-1, j-1], self.terrain[i-1, j+1],
                        self.terrain[i+1, j-1], self.terrain[i+1, j+1]
                    ]
                    if self.terrain[i, j] <= min(neighbors):
                        valleys.append((i, j))
        return valleys
    
    def print_analysis(self):
        """Print detailed terrain analysis"""
        stats = self.calculate_statistics()
        peaks = self.find_peaks()
        valleys = self.find_valleys()
        
        print("=" * 60)
        print("TERRAIN ANALYSIS REPORT")
        print("=" * 60)
        print(f"\nStatistical Properties:")
        print(f"  Mean Height:     {stats['mean_height']:.4f}")
        print(f"  Max Height:      {stats['max_height']:.4f}")
        print(f"  Min Height:      {stats['min_height']:.4f}")
        print(f"  Std Deviation:   {stats['std_dev']:.4f}")
        print(f"  Roughness Index: {stats['roughness']:.4f}")
        
        print(f"\nTerrain Features:")
        print(f"  Number of Peaks:   {len(peaks)}")
        print(f"  Number of Valleys: {len(valleys)}")
        
        if peaks:
            print(f"\n  Top 5 Highest Peaks:")
            peak_heights = [(p, self.terrain[p]) for p in peaks]
            peak_heights.sort(key=lambda x: x[1], reverse=True)
            for idx, (pos, height) in enumerate(peak_heights[:5], 1):
                print(f"    {idx}. Position {pos}, Height: {height:.4f}")
        
        print("=" * 60)

def main():
    """Main execution function"""
    print("🏔️  Procedural Terrain Generator 🏔️\n")
    print("Generating terrain...")
    
    # Create terrain with custom configuration
    config = TerrainConfig(
        size=200,
        octaves=6,
        persistence=0.5,
        lacunarity=2.0,
        scale=100.0,
        seed=42  # Use None for random seed
    )
    
    # Generate terrain
    generator = TerrainGenerator(config)
    terrain = generator.generate_heightmap()
    print("✓ Base terrain generated")
    
    # Add features
    generator.add_features()
    print("✓ Terrain features added")
    
    # Apply erosion
    generator.apply_erosion(iterations=3)
    print("✓ Erosion simulation applied")
    
    # Analyze terrain
    analyzer = TerrainAnalyzer(generator.terrain)
    analyzer.print_analysis()
    
    # Visualize terrain
    print("\nCreating visualizations...")
    visualizer = TerrainVisualizer(generator.terrain)
    
    # Create 3D plot
    fig1, ax1 = visualizer.plot_3d(elev=35, azim=45)
    print("✓ 3D visualization created")
    
    # Create topographic map
    fig2 = visualizer.plot_topographic()
    print("✓ Topographic map created")
    
    # Create shaded relief
    fig3 = visualizer.plot_shaded_relief()
    print("✓ Shaded relief map created")
    
    print("\n🎨 All visualizations complete!")
    print("Close the windows to exit.")
    
    plt.show()

if __name__ == "__main__":
    main()