from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np

def get_terrain_cmap():
    c0 = '#2351b7'
    c1 = '#1175db'
    c2 = '#00acc4'
    c3 = '#0fce68'
    c4 = '#fefd98'
    c5 = 'peru'
    c6 = 'sienna'
    c7 = 'maroon'
    c8 = 'snow'

    colors = [
        (0.0, c0),
        (0.05, c1),
        (0.1, c2),
        (0.22, c3),
        (0.45, c4),
        (0.65, c5),
        (0.8, c6),
        (1.0,  c7),
    ]

    cmap = LinearSegmentedColormap.from_list('white_terrain', colors=colors, N=256)
    return cmap
    
def get_wave_cmap():
    """Create a custom colormap with smooth transitions between the given colors."""
    c0 = 'darkslateblue'
    c1 = 'royalblue'
    c2 = 'cornflowerblue'
    #c2 = 'lightblue'
    c3 = 'lavender'
    c4 = 'white' # 'whitesmoke'
    c5 = 'palegoldenrod'
    c55 = '#EEE600'
    c6 = 'goldenrod'
    c7 = 'indianred' # firebrick
    c8 = 'darkred'

    # colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.55, c5), (0.65, c55), (0.8, c6), (0.9, c7), (1.0, c8)]
    colors = [(0.0, c0), (0.1, c1), (0.2, c2), (0.4, c3), (0.48, c4), (0.52, c4), (0.6, c5), (0.75, c6), (0.85, c7), (1.0, c8)]
    # colors = [(0.0, c0), (0.32, c0), (0.4, c1), (0.6, c1), (0.68, c2), (0.73, c2), (0.78, c3), (1.0, c3)]
    
    cmap = LinearSegmentedColormap.from_list('wave', colors=colors, N=256)
    return cmap

def get_spectral_white_cmap():
    """Return a modified 'Spectral' colormap with white replacing the yellow center."""
    # Sample colors from the original 'Spectral' colormap
    base = plt.get_cmap('Spectral')
    
    # Extract colors and manually replace midrange with white
    n = 256
    colors = base(np.linspace(0, 1, n))
    
    # Define a "white region" around the center (original is bright yellow there)
    mid_low, mid_high = int(n * 0.46), int(n * 0.54)
    for i in range(mid_low, mid_high):
        # Blend softly into white instead of abrupt cutoff
        blend = (i - mid_low) / (mid_high - mid_low)
        colors[i, :3] = [1, 1, 1]  # pure white RGB
        colors[i, 3] = 1.0         # full alpha
    
    # Build new colormap
    cmap = LinearSegmentedColormap.from_list('spectral_white', colors, N=n)
    return cmap
    
def get_greenpurple_cmap():
    """Diverging colormap: blue → green → white → yellow/red, keeping positive side identical to get_wave_cmap()."""
    
    # Negative side
    c0 = 'darkslategrey'
    c1 = 'teal'
    c2 = 'seagreen'
    c3 = 'mediumseagreen'
    
    # Center
    c4 = 'white'

    c5 = 'mistyrose'
    c6 = 'peachpuff'
    c7 = 'plum' # firebrick
    c8 = 'rebeccapurple'

    colors = [(0.0, c0),
              (0.15, c2),
              (0.3, c3),
              (0.48, c4),
              (0.52, c4),
              (0.6, c5),
              # (0.7, c6),
              (0.8, c7),
              (1.0, c8)]
    
    cmap = LinearSegmentedColormap.from_list('wave_bluegreen', colors=colors, N=256)
    return cmap
    

def get_coolwarm_soft_cmap():
    """Elegant soft diverging colormap from turquoise to rose with white center."""
    c0 = '#006d77'   # deep teal
    c1 = '#83c5be'   # light teal
    c2 = '#edf6f9'   # pale cyan
    c3 = 'white'
    c4 = '#ffe5ec'   # pale pink
    c5 = '#f4a7b9'   # muted rose
    c6 = '#9b2226'   # deep red-brown

    colors = [
        (0.0, c0), (0.2, c1), (0.35, c2),
        (0.48, c3), (0.52, c3),
        (0.65, c4), (0.8, c5), (1.0, c6)
    ]
    return LinearSegmentedColormap.from_list('coolwarm_soft', colors=colors, N=256)

def get_purplegold_cmap():
    """Diverging colormap with purples and golds centered on white."""
    c0 = '#3b0a45'   # deep violet
    c1 = '#7b3294'   # purple
    c2 = '#c2a5cf'   # lilac
    c3 = 'white'
    c4 = '#fddbc7'   # soft beige
    c5 = '#f4a582'   # salmon-gold
    c6 = '#b2182b'   # dark red

    colors = [
        (0.0, c0), (0.2, c1), (0.35, c2),
        (0.48, c3), (0.52, c3),
        (0.65, c4), (0.8, c5), (1.0, c6)
    ]
    return LinearSegmentedColormap.from_list('purplegold', colors=colors, N=256)