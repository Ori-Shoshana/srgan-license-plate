"""Desktop GUI: pick an image and see the super-resolved plate and the detected text."""

import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk

import pipeline


class ImageSelectorApp:
    def __init__(self, master):
        self.master = master
        self.master.title("License Plate Super-Resolution")

        self.image_label = tk.Label(self.master)
        self.image_label.pack()

        self.select_button = tk.Button(self.master, text="Select Image", command=self.load_image)
        self.select_button.pack()

        self.selected_image = None

    def load_image(self):
        file_path = filedialog.askopenfilename()
        if file_path:
            self.selected_image = Image.fromarray(pipeline.project(file_path))
            self.selected_image = self.selected_image.resize((1000, 1000))
            self.display_image()

    def display_image(self):
        if self.selected_image:
            photo_image = ImageTk.PhotoImage(self.selected_image)
            self.image_label.config(image=photo_image)
            self.image_label.image = photo_image  # keep a reference


def main():
    root = tk.Tk()
    ImageSelectorApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
