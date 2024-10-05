import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os  
    
def plot_spectra(input_file, output_dir, x_val, y_val, waves, raw, denoised, smoothed_intensities, baseline, corrected_intensities, id_fit, ig_fit, id2_fit, ig2_fit, combined_fit):
    plt.figure(figsize=(12, 8))
    
    # Plot raw intensity
    plt.plot(waves, raw, label='Raw Data', color='black')
    
    # Plot denoised intensity
    plt.plot(waves, denoised, label='Denoised', color='red')
    
    # Plot smoothed intensity
    plt.plot(waves, smoothed_intensities, label='Smoothed', color='blue')
    
    # Plot baseline
    plt.plot(waves, baseline, label='Baseline', color='yellow', linestyle='--')
    
    # Plot baseline-corrected intensity
    plt.plot(waves, corrected_intensities, label='Baseline Corrected', color='black', linewidth=2)
    
    # Plot fitting results
    plt.plot(waves, id_fit, label='D Band Fit', color='green', linestyle=':')
    plt.plot(waves, ig_fit, label='G Band Fit', color='pink', linestyle=':')
    plt.plot(waves, id2_fit, label='D2 Band Fit', color='purple', linestyle=':')
    plt.plot(waves, ig2_fit, label='G2 Band Fit', color='brown', linestyle=':')
    plt.plot(waves, combined_fit, label='Combined Fit', color='red', linewidth=1)
        
    plt.xlabel('Wavenumber (cm^-1)')
    plt.xlim(1000,1800)
    plt.ylabel('Intensity (a.u.)')
    plt.yticks([])
    plt.title(f'{input_file} Spectra: ({x_val},{y_val})')
    plt.legend()
    plt.grid(True)
        
    # Save plot
    plot_file = os.path.join(output_dir, f'Fitted-Spectra_{x_val},{y_val}.png')
    plt.savefig(plot_file)
    plt.close()

def plot_heatmap(title, data, file_name, output_dir, max_x, max_y, v_min, v_max):
    plt.figure(figsize=(2, 20))
    ax = sns.heatmap(
        data,
        cmap='viridis',
        annot=False,  # Remove printed values
        vmin=v_min,  # Minimum value of the colorbar
        vmax=v_max,  # Maximum value of the colorbar
        cbar_kws={'label': 'ID/IG Ratio Colour Scale', 'orientation': 'vertical'}
    )

    # Access the colorbar and configure it
    cbar = ax.collections[0].colorbar
    cbar.set_label('ID/IG Ratio Colour Scale', fontsize=25, fontweight='bold')
    
    # Set colorbar ticks and labels
    num_ticks = max(10, int((v_max - v_min) / 0.1))  # Ensure at least 2 ticks, and use a reasonable interval
    tick_values = np.linspace(v_min, v_max, num=num_ticks)
    cbar.set_ticks(tick_values)
    cbar.ax.set_yticklabels([f'{label:.2f}' for label in tick_values])
    cbar.ax.tick_params(labelsize=25)

    # Set x and y limits to match the size of the matrix
    ax.set_xlim(0, max_y)
    ax.set_ylim(max_x, 0)

    # Adjust the colorbar size and position to stretch the entire length of the heatmap
    pos = ax.get_position()
    cbar_width = 0.8  # Width of the colorbar
    cbar_padding = 0.2  # Padding between the heatmap and colorbar
    cbar_position = [
        pos.x1 + cbar_padding,  # Left position
        pos.y0,  # Bottom position
        cbar_width,  # Width
        pos.height  # Height (stretch to match heatmap height)
    ]
    cbar.ax.set_position(cbar_position)

    # Centre title
    plt.title(f"Heatmap of {file_name} ID/IG {title} Ratio", fontsize=25, fontweight='bold', loc='center', y=1.02)

    # Increase the font size of the labels and markers
    plt.xlabel('X', fontsize=25, fontweight='bold')
    plt.ylabel('Y', fontsize=25, fontweight='bold')
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    ax.xaxis.set_major_locator(plt.MultipleLocator(1))  # X-axis tick intervals of 1
    ax.yaxis.set_major_locator(plt.MultipleLocator(5))  # Y-axis tick intervals of 5

    # Set the tick positions and labels
    xticks = np.arange(0, max_y + 1, 1)
    yticks = np.arange(0, max_x + 1, 5)
    ax.invert_yaxis()
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.set_xticklabels([f'{i:.0f}' for i in xticks])
    ax.set_yticklabels([f'{i:.1f}' for i in yticks])
    ax.tick_params(axis='both', which='major', labelsize=25)

    # Save the figure with the colorbar aligned to the heatmap
    output_file = os.path.join(output_dir, f"{file_name}-ID-IG-{title}-Map.png")
    plt.savefig(output_file, dpi=300, bbox_inches='tight', pad_inches=0.1)
