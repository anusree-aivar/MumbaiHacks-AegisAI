from PIL import Image, ImageDraw, ImageOps
import math

def create_marquee_gif(input_path, output_path):
    # --- Configuration ---
    TARGET_HEIGHT = 80       # Height of the logo in the banner
    GAP = 60                 # Space between logos
    VIEWPORT_WIDTH = 600     # Width of the final GIF
    BG_COLOR = (20, 20, 20, 255) # Dark background (to make white logo visible)
    SPEED = 4                # Higher = faster, Lower = smoother/slower
    FADE_WIDTH = 100         # Width of the gradient fade on edges

    # 1. Load and Clean Image
    img = Image.open(input_path).convert("RGBA")
    
    # Trim whitespace
    bbox = img.getbbox()
    if bbox:
        img = img.crop(bbox)
    
    # Resize maintaining aspect ratio
    aspect_ratio = img.width / img.height
    new_width = int(TARGET_HEIGHT * aspect_ratio)
    img = img.resize((new_width, TARGET_HEIGHT), Image.Resampling.LANCZOS)

    # 2. Create the Infinite Strip
    # We need enough copies to fill the viewport + one full cycle for smooth looping
    unit_width = new_width + GAP
    num_copies = math.ceil(VIEWPORT_WIDTH / unit_width) + 2
    
    strip_width = num_copies * unit_width
    strip_img = Image.new("RGBA", (strip_width, TARGET_HEIGHT), (0, 0, 0, 0))
    
    for i in range(num_copies):
        strip_img.paste(img, (i * unit_width, 0), img)

    # 3. Create the Alpha Mask (The "Fade" Effect)
    # This creates a gradient: Transparent -> Opaque -> Transparent
    mask = Image.new("L", (VIEWPORT_WIDTH, TARGET_HEIGHT), 255)
    draw = ImageDraw.Draw(mask)
    
    # Draw Left Fade
    for x in range(FADE_WIDTH):
        opacity = int(255 * (x / FADE_WIDTH))
        draw.line([(x, 0), (x, TARGET_HEIGHT)], fill=opacity)
        
    # Draw Right Fade
    for x in range(FADE_WIDTH):
        opacity = int(255 * (x / FADE_WIDTH))
        draw.line([(VIEWPORT_WIDTH - 1 - x, 0), (VIEWPORT_WIDTH - 1 - x, TARGET_HEIGHT)], fill=opacity)

    # 4. Generate Frames
    frames = []
    # We only need to move exactly one 'unit_width' (logo + gap) to create a perfect loop
    total_frames = range(0, unit_width, SPEED)
    
    for x_offset in total_frames:
        # Create background
        frame = Image.new("RGBA", (VIEWPORT_WIDTH, TARGET_HEIGHT), BG_COLOR)
        
        # Crop the specific section of the strip
        # We shift the crop box to the right to simulate movement to the left
        crop_box = (x_offset, 0, x_offset + VIEWPORT_WIDTH, TARGET_HEIGHT)
        visible_strip = strip_img.crop(crop_box)
        
        # Paste the strip onto the frame
        frame.paste(visible_strip, (0, 0), visible_strip)
        
        # Apply the gradient mask to the alpha channel of the frame
        # (This ensures the background color stays, but the content fades if we wanted transparency)
        # However, for a solid looking gif, we composite the fade on top.
        # Let's apply the mask to the result to fade it into the BG_COLOR if needed,
        # or simply put the fading overlay on top. 
        # A simpler way for GIF quality is to composite the frame.
        
        final_frame = frame.copy()
        final_frame.putalpha(mask)
        
        # Since GIF doesn't do partial transparency well, we composite against the background color
        # to "bake in" the fade
        bg_base = Image.new("RGBA", (VIEWPORT_WIDTH, TARGET_HEIGHT), BG_COLOR)
        bg_base.paste(final_frame, (0, 0), final_frame)
        
        frames.append(bg_base)

    # 5. Save GIF
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        optimize=False,
        duration=40, # ms per frame
        loop=0
    )

# Usage
create_marquee_gif("/Users/dhanush/Downloads/new copy/backend/logo/aivar-new.png", "aivar_marquee.gif")