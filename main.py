# main.py
from __future__ import annotations
import os
import sys
import tkinter as tk
import tkinter.filedialog as fd

# (Optional) silence mac Tk deprecation chatter
os.environ.setdefault("TK_SILENCE_DEPRECATION", "1")

# IMPORTANT: set MPL backend before importing pyplot anywhere
import matplotlib
matplotlib.use("TkAgg")

import cv2
from GUI import GUI

def choose_image_path(initialdir: str | None = None) -> str:
    # Very permissive filters (includes uppercase extensions)
    img_exts = (".png",".jpg",".jpeg",".bmp",".tif",".tiff",
                ".PNG",".JPG",".JPEG",".BMP",".TIF",".TIFF")
    filetypes = [
        ("Image files", img_exts),
        ("All files", "*"),
    ]
    if initialdir is None:
        initialdir = os.path.expanduser("~/Pictures")
        if not os.path.isdir(initialdir):
            initialdir = os.path.expanduser("~")

    # One root; keep it hidden
    root = tk.Tk()
    root.withdraw()
    root.update()  # ensure Tk is initialized on main thread

    path = fd.askopenfilename(
        title="Select bearing image",
        initialdir=initialdir,
        filetypes=filetypes,
    )

    root.destroy()
    return path

def main():
    path = choose_image_path()
    if not path:
        print("No image selected. Exiting.")
        sys.exit(0)

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Could not read image: {path}")
        sys.exit(1)

    gui = GUI(img)
    tk.mainloop()

if __name__ == "__main__":
    main()
