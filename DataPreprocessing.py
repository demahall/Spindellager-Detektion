import tkinter as tk
from tkinter.filedialog import askopenfilename
import cv2

root = tk.Tk()
root.withdraw()  # hide the root window

path = askopenfilename(title="Select bearing image",
                       filetypes=[("Images","*.png;*.jpg;*.jpeg;*.bmp;*.tif;*.tiff")])
if not path:
    raise SystemExit("No image selected.")
img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
