import imageio
import numpy as np
import tempfile

def get_video(frames, filename, fps=7):

    with imageio.get_writer(filename, fps=fps) as writer:
        for image in frames:
            writer.append_data(np.asarray(image))
    
    return filename