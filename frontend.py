from tkinter import *
import backend_city
import backend_zoo

def temp():
    print("Hello")


window = Tk()

window.wm_title("Natural Disaster Predictor and Detector")

l1 = Label(window, text = "Video File Path")
l1.grid(row = 0, column = 0)

video_path = StringVar()
e1 = Entry(window, textvariable = video_path)
e1.grid(row = 1, column = 1)

b1 = Button( window, text = "Detector", width = 12, command = lambda : backend_city.detect_discrepancy(video_path.get()) )
b1.grid(row = 3, column = 2)

b2 = Button(window, text = "Zoo Surveilance", command = lambda: backend_zoo.entry_exit(video_path.get()))
b2.grid(row = 3, column = 3)

window.mainloop()
