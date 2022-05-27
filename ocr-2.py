from tkinter.constants import NW
from tkinter.filedialog import askopenfilename
import tkinter as tk
from PIL import Image, ImageTk


class App(tk.Tk):
    def __init__(self):
        tk.Tk.__init__(self)
        self.x = self.y = 0
        # Creating elements
        self.canvas = tk.Canvas(
            self, width=300, height=300, bg="black")

        self.label = tk.Label(self, text="Thinking..", font=("Helvetica", 48))

        self.classify_btn = tk.Button(
            self, text="Recognise", command=self.classify_handwriting)

        self.button_import = tk.Button(
            self, text="Select image", command=self.import_image)

        # Grid structure
        self.canvas.grid(row=0, column=0, pady=2)
        self.label.grid(row=0, column=1, pady=2, padx=2)
        self.classify_btn.grid(row=1, column=1, pady=2, padx=2)
        self.button_import.grid(row=1, column=0, pady=2)

    def clear_all(self):
        self.canvas.delete("all")

    def classify_handwriting(self):
        digit, acc = predict_digit(im)
        self.label.configure(text=str(digit)+', ' + str(int(acc*100))+'%')

    def import_image(self):
        name = askopenfilename(initialdir="/",
                               filetypes=(("PNG File", "*.png"), ("BMP File",
                                                                  "*.bmp"), ("JPEG File", "*.jpeg")),
                               title="Choose a file."
                               )

        img = Image.open(name)
        img = img.resize((250, 250))
        tkimage = ImageTk.PhotoImage(img)

        # pack the canvas into a frame/form
        self.canvas.pack()
        self.canvas.create_image(image=tkimage, anchor=NW)

    def draw_lines(self, event):
        self.x = event.x
        self.y = event.y
        r = 8
        self.canvas.create_oval(
            self.x-r, self.y-r, self.x + r, self.y + r, fill='black')


app = App()
mainloop()
