# Spatial Grid Configuration
WIDTH_2D = 400
HEIGHT_2D = 400
WIDTH_3D = 100
HEIGHT_3D = 100
DEPTH_3D = 100
MAX_STEPS = 150
SEED = 43
INITIAL_CELLS = 1

# Microenvironment Parameters (Physics)
OXYGEN_BG = 1.0      # Background oxygen concentration (c0) 
GLUCOSE_BG = 1.0     # Background glucose concentration (g0)
D_C = 0.1            # Oxygen diffusion coefficient [cite: 437]
D_G = 0.1            # Glucose diffusion coefficient
D_H = 0.1            # H+ ions diffusion coefficient
H_IONS_BG = 0.0      # Background H+ concentration (h0)

# Biological Cell Parameters (Genetics)
MUTATION_PROB = 0.01 # Mutation probability per gene (p) [cite: 421]
MUTATION_STD = 0.25  # Mutation standard deviation (sigma) [cite: 428]
BASE_O2_RATE = 0.03  # Base oxygen consumption rate [cite: 312, 387]