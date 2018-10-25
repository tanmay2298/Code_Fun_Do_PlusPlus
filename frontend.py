from tkinter import *
from PIL import ImageTk, Image
import backend_city
import backend_zoo

window = Tk()

window.wm_title("Natural Disaster Predictor and Detector")

l1 = Label(window, text = "Video File Path")
l1.grid(row = 3, column = 4)

video_path = StringVar()
e1 = Entry(window, textvariable = video_path)
e1.grid(row = 4, column = 4)


b1 = Button(window, text = "Detector", width = 12, command = lambda : backend_city.detect_discrepancy(video_path.get()))
b1.grid(row = 6, column = 3)


b2 = Button(window, text = "Zoo Surveilance", command = lambda: backend_zoo.entry_exit(video_path.get()))
b2.grid(row = 6, column = 5)

city_image = ImageTk.PhotoImage(Image.open("city.jpeg"))
img1 = Label(window, image = city_image).grid(row = 7, column = 3)

zoo_image = ImageTk.PhotoImage(Image.open("zoo.jpeg"))
img2 = Label(window, image = zoo_image).grid(row = 7, column = 5)

window.mainloop()
