import os
import sys
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
# Simple 3D grid class to manage single occupancy of voxels by agent cells
class SingleGrid3D:
    """
    Minimal 3D grid implementation for Mesa-like agent placement.
    """
    def __init__(self, width, height, depth):
        """
        Initializes the 3D grid with specific dimensions.
        """
        self.width = width
        self.height = height
        self.depth = depth
        self.grid = np.empty((width, height, depth), dtype=object)

    # Place an agent at a specific coordinate
    def place_agent(self, agent, pos):
        """
        Assigns an agent to a specific (x, y, z) voxel.
        """
        x, y, z = pos
        self.grid[x, y, z] = agent
        agent.pos = pos

    # Remove an agent from the grid
    def remove_agent(self, agent):
        """
        Clears an agent from its current voxel.
        """
        if agent.pos is None:
            return
        x, y, z = agent.pos
        self.grid[x, y, z] = None
        agent.pos = None

    # Check if a voxel is unoccupied
    def is_cell_empty(self, pos):
        """
        Returns True if the voxel at pos is empty.
        """
        x, y, z = pos
        return self.grid[x, y, z] is None

    # Find coordinate neighbors using Moore (26-neighbors) or Von Neumann (6-neighbors)
    def get_neighborhood(self, pos, moore=False, include_center=False):
        """
        Returns a list of adjacent coordinates in 3D.
        """
        x, y, z = pos
        neighbors = []
        if moore:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dz in (-1, 0, 1):
                        if not include_center and dx == dy == dz == 0:
                            continue
                        nx, ny, nz = x + dx, y + dy, z + dz
                        if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                            neighbors.append((nx, ny, nz))
        else:
            candidates = [
                (x - 1, y, z),
                (x + 1, y, z),
                (x, y - 1, z),
                (x, y + 1, z),
                (x, y, z - 1),
                (x, y, z + 1),
            ]
            for nx, ny, nz in candidates:
                if 0 <= nx < self.width and 0 <= ny < self.height and 0 <= nz < self.depth:
                    neighbors.append((nx, ny, nz))
        if include_center:
            neighbors.append((x, y, z))
        return neighbors

    # Return agent objects residing in the neighborhood
    def get_neighbors(self, pos, moore=False, include_center=False):
        positions = self.get_neighborhood(pos, moore=moore, include_center=include_center)
        return [self.grid[x, y, z] for x, y, z in positions if self.grid[x, y, z] is not None]
