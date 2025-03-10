import torch
import vispy.scene
from vispy.scene import visuals
import sys

    

def initialize_grid(grid_size):
    """Creates a random initial grid on the GPU with a 5% chance of live cells."""
    return (torch.rand(grid_size, device=device) < 0.05).float()

def count_neighbors(grid):
    """Counts live neighbors for each cell using toroidal wrapping."""
    neighbors = (
        grid.roll(1, dims=0).roll(1, dims=1) +  # Top-left
        grid.roll(1, dims=0) +  # Top
        grid.roll(1, dims=0).roll(-1, dims=1) +  # Top-right
        grid.roll(-1, dims=1) +  # Right
        grid.roll(1, dims=1) +  # Left
        grid.roll(-1, dims=0).roll(-1, dims=1) +  # Bottom-right
        grid.roll(-1, dims=0) +  # Bottom
        grid.roll(-1, dims=0).roll(1, dims=1)   # Bottom-left
    )
    return neighbors

def update(grid):
    """Applies the Game of Life rules."""
    neighbors = count_neighbors(grid)
    new_grid = (neighbors == 3) | ((grid == 1) & (neighbors == 2))
    return new_grid.to(torch.float32)

def main():
    """Main function to initialize and run the Game of Life simulation."""
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Default grid size
    grid_size = (200, 200)

    # Check for command-line arguments
    if len(sys.argv) == 3:
        try:
            grid_size = (int(sys.argv[1]), int(sys.argv[2]))
        except ValueError:
            print("Invalid grid size. Using default (200, 200).")
    
    print(f"Starting Game of Life with grid size: {grid_size}")

    # Initialize the game grid on GPU
    grid = initialize_grid(grid_size)

    # Set up VisPy visualization
    canvas = vispy.scene.SceneCanvas(keys='interactive', show=True, bgcolor='black', size=(800, 800))
    view = canvas.central_widget.add_view()

    # Configure camera settings
    view.camera = 'panzoom'  # Enable zoom & panning
    view.camera.aspect = 1  # Maintain aspect ratio
    view.camera.set_range(x=(0, grid_size[1]), y=(0, grid_size[0]))  # Fit grid in view

    # Create the image with proper scaling
    image = visuals.Image(grid.cpu().numpy(), cmap="magma", interpolation="nearest", clim=(0, 1))
    view.add(image)

    def update_frame(event):
        """Updates the grid and refreshes the visualization directly on GPU."""
        nonlocal grid
        grid = update(grid)  # Perform update on GPU

        # Normalize grid values for better visibility
        img_data = grid.cpu().numpy()
        img_data = (img_data - img_data.min()) / (img_data.max() - img_data.min() + 1e-8)  # Normalize to (0,1)

        image.set_data(img_data)  # Update visualization
        canvas.update()  # Refresh VisPy canvas

    # Use VisPy Timer for smooth animation
    timer = vispy.app.Timer(interval=0.1, connect=update_frame, start=True)

    # Run VisPy event loop
    vispy.app.run()

if __name__ == "__main__":
    main()
