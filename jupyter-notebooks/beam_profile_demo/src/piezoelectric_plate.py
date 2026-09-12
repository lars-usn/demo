import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

FIGURE_NAME = "Piezoelectric Plate Demo"


def draw_voltmeter():
    """Draw voltmeter with pointer."""
    voltmeter_circle = plt.Circle(
        (voltmeter_x, voltmeter_y),
        voltmeter_radius,
        color='#f0f4f8',
        ec='black',
        lw=2.5,
        zorder=3)

    ax.add_patch(voltmeter_circle)

    scale_angle = np.radians(np.linspace(225, -45, 40))
    scale_radius = 0.8 * voltmeter_radius

    scale_x = voltmeter_x + scale_radius * np.cos(scale_angle)
    scale_y = voltmeter_y + scale_radius * np.sin(scale_angle)
    ax.plot(
        scale_x,
        scale_y,
        color='black',
        lw=1.5,
        zorder=4)

    for rad in scale_angle[::3]:
        tx_start = voltmeter_x + (scale_radius - 0.08) * np.cos(rad)
        ty_start = voltmeter_y + (scale_radius - 0.08) * np.sin(rad)
        tx_end = voltmeter_x + (scale_radius + 0.05) * np.cos(rad)
        ty_end = voltmeter_y + (scale_radius + 0.05) * np.sin(rad)

        ax.plot(
            [tx_start, tx_end],
            [ty_start, ty_end],
            color='black',
            lw=1.5,
            zorder=4,
        )

    pointer_length = 0.95 * scale_radius
    pointer_angle = np.radians(voltmeter_value)
    pointer_x = voltmeter_x + pointer_length * np.cos(pointer_angle)
    pointer_y = voltmeter_y + pointer_length * np.sin(pointer_angle)

    ax.annotate('',
                xy=(pointer_x, pointer_y),
                xytext=(voltmeter_x, voltmeter_y),
                arrowprops=dict(
                    arrowstyle="->",
                    lw=6,
                    color='crimson',
                    mutation_scale=15),
                zorder=5)

    center_marker = plt.Circle(
        (voltmeter_x, voltmeter_y),
        0.08,
        color='black',
        zorder=6
    )
    ax.add_patch(center_marker)


fig, ax = plt.subplots(
    figsize=(12, 8),
    layout="constrained",
    num=FIGURE_NAME,
)


# --- Main progrem ------------------------------------------------
# Sizes and positions
voltmeter_radius = 1.0
voltmeter_x = 2.0
voltmeter_y = 0
voltmeter_value = 90

plate_x = 6.0
plate_y = 0.0
plate_width = 4.0
plate_thickness = 2.0

plate_top = plate_y + plate_thickness / 2
plate_bottom = plate_y - plate_thickness / 2

draw_voltmeter()

# --- 4. TEGN DEN GRÅBRUNE PLATEN ---
grabrune_plate = patches.Rectangle((plate_x, plate_y), plate_width, plate_thickness,
                                   facecolor='#8B8580', edgecolor='black', lw=2, zorder=3)
ax.add_patch(grabrune_plate)

# Tekst på platen
ax.text(plate_x + plate_width/2, plate_y + plate_thickness/2, 'Piezoelectric plate',
        fontsize=14, fontweight='bold', color='white', ha='center', va='center', zorder=4)


# --- 5. TEGN LEDNINGENE (TIL VENSTRE HJØRNER AV PLATA) ---
# Rød ledning fra øvre venstre hjørne til toppen av voltmeteret
rod_x = [plate_x, plate_x - 1.0,
         plate_x - 1.0, voltmeter_x, voltmeter_x]
rod_y = [plate_top, plate_top, plate_top +
         1.0, plate_top + 1.0, voltmeter_y + voltmeter_radius]
ax.plot(rod_x, rod_y, color='red', lw=3, label='Øvre ledning')

# Blå ledning fra nedre venstre hjørne til bunnen av voltmeteret
bla_x = [plate_x, plate_x - 1.0, voltmeter_x, voltmeter_x]
bla_y = [plate_bottom, plate_bottom -
         1.0, plate_bottom - 1.0, voltmeter_y - voltmeter_radius]
ax.plot(bla_x, bla_y, color='blue', lw=3, label='Nedre ledning')

# Koblingspunkter (på voltmeteret og på hjørnene av platen)
ax.scatter([voltmeter_x, voltmeter_x], [voltmeter_y + voltmeter_radius, voltmeter_y - voltmeter_radius],
           color='black', s=60, zorder=5)
ax.scatter([plate_x, plate_x], [plate_top,
           plate_bottom], color='black', s=50, zorder=5)


# --- 6. TEGN DE VERTIKALE PILENE ---
midt_x = plate_x + (plate_width / 2)

# Pil over platen som peker NEDOVER mot toppen (fra y=5.0 ned til y=plate_y + plate_thickness + 0.1)
ax.annotate('', xy=(midt_x, plate_y + plate_thickness), xytext=(midt_x, 6.0),
            arrowprops=dict(facecolor='darkred', edgecolor='darkred',
                            width=6, headwidth=12, shrink=0.05),
            zorder=4)

# Pil under platen som peker UPPOVER mot bunnen (fra y=2.0 opp til y=plate_y - 0.1)
ax.annotate('', xy=(midt_x, plate_y - 0.1), xytext=(midt_x, 2.0),
            arrowprops=dict(facecolor='darkblue', edgecolor='darkblue',
                            width=4, headwidth=12, shrink=0.05),
            zorder=4)


# --- 7. FORMATERING AV DIAGRAMMET ---
ax.set_title('Voltmeter koblet til plate med retningspiler',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 12)
ax.set_ylim(-4, 4)
ax.legend(loc='upper right')

ax.axis('on')

plt.tight_layout()
plt.show()
