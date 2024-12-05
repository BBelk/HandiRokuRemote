import tkinter as tk
from PIL import Image, ImageTk


class Tooltip:
    def __init__(self, widget, image_path=None, text=None, wide=None):
        self.widget = widget
        self.wide = wide
        self.image_path = image_path
        self.text = text
        self.tooltip_window = None
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.hide_tooltip)

    def show_tooltip(self, event):
        if self.tooltip_window:
            return  # Tooltip already displayed

        # Create a top-level window for the tooltip
        self.tooltip_window = tk.Toplevel(self.widget)
        self.tooltip_window.wm_overrideredirect(True)  # Remove window decorations

        # Get the position of the widget to place the tooltip near it
        x, y, _, _ = self.widget.bbox("insert")
        x += self.widget.winfo_rootx() + 20  # Offset for better positioning
        y += self.widget.winfo_rooty() + 20
        self.tooltip_window.geometry(f"+{x}+{y}")

        # Add image or text to the tooltip
        if self.image_path:
            img = Image.open(self.image_path)
            width = 100
            height = 100
            if self.wide:
                width *= 2  # Double the width if `wide` is True
            img = img.resize((width, height))  # Resize for consistency
            self.tooltip_image = ImageTk.PhotoImage(img)  # Store reference to avoid garbage collection
            label = tk.Label(self.tooltip_window, image=self.tooltip_image, borderwidth=1, relief="solid")
        else:
            label = tk.Label(self.tooltip_window, text=self.text, borderwidth=1, relief="solid")

        label.pack()

    def hide_tooltip(self, event):
        if self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None
