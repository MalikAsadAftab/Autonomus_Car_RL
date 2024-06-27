import numpy as np

def setup_game(size, d, entities, layers=1):
    """
    Initialize the game environment with optional layers.
    
    Args:
        size (int): The size of the environment (width and height).
        d (dict): A dictionary mapping entity types to color values.
        entities (list of tuples): Each tuple contains (entity_type, x, y).
        layers (int): The number of layers in the environment.
    
    Returns:
        numpy.ndarray: The initialized environment.
    """
    # Define environment based on whether it's multi-layered
    if layers > 1:
        env = np.zeros((size, size, 3, layers), dtype=np.uint8)
    else:
        env = np.zeros((size, size, 3), dtype=np.uint8)
    
    for entity_type, x, y in entities:
        if entity_type in d:
            color = d[entity_type]
            if layers > 1:
                for layer in range(layers):
                    env[x, y, :, layer] = color
            else:
                env[x, y] = color  # Properly handle the single-layer case
    
    return env
