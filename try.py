import tkinter as tk
import time
import math
import random

rate = 20

class CAR(object):
    __slots__ = ("__length","__width","v","Way","X","Y","alpha","tha","car")

    def __init__(self,comeFrom,goTo):
        self.__length = 50
        self.__width = 20
        self.v = 4
        self.Way = (comeFrom,goTo)
        mat = ((550,265), (265,50),(50,335),(335,550))
        self.X,self.Y = mat[comeFrom][0],mat[comeFrom][1]
        if 230 < self.Y < 370:self.alpha = (1,0)
        else:self.alpha = (0,1)
        self.tha = 0


    def GO(self):
        if 230 < self.X < 370 and 230 < self.Y < 370:
            if self.Way[1]-self.Way[0] == 1 or (self.Way[1] == 0 and self.Way[0] == 3):
                leftO = ((370,230,270),(230,230,0),(230,370,90),(370,370,180))
                self.tha += 10
                self.X,self.Y = math.ceil(leftO[self.Way[0]][0]+35*math.cos((leftO[self.Way[0]][2]-self.tha)*3.14/180)),\
                                math.ceil(leftO[self.Way[0]][1]-35*math.sin((leftO[self.Way[0]][2]-self.tha)*3.14/180))
                if self.Way[0]%2 == 0: self.alpha = (math.cos(self.tha*3.14/180),math.sin(self.tha*3.14/180))
                else:self.alpha = (math.sin(self.tha*3.14/180),-math.cos(self.tha*3.14/180))

            elif self.Way[0]-self.Way[1] == 1 or (self.Way[0] == 0 and self.Way[1] == 3):
                leftO = ((370,370,90),(370,230,180),(230,230,270),(230,370,0))
                self.tha += 10
                self.X,self.Y = math.ceil(leftO[self.Way[0]][0]+105*math.cos((leftO[self.Way[0]][2]+self.tha)*3.14/180)),\
                                math.ceil(leftO[self.Way[0]][1]-105*math.sin((leftO[self.Way[0]][2]+self.tha)*3.14/180))
                if self.Way[0]%2 == 0: self.alpha = (math.cos(self.tha*3.14/180),-math.sin(self.tha*3.14/180))
                else:self.alpha = (math.sin(self.tha*3.14/180),math.cos(self.tha*3.14/180))
            else:
                if self.Way[1] == 0:
                    self.X += self.v
                elif self.Way[1] == 1:
                    self.Y -= self.v
                elif self.Way[1] == 2:
                    self.X -= self.v
                else:
                    self.Y += self.v
        else:
            if 240 <= self.Y <= 290:
                self.X -= self.v
                self.alpha = (1, 0)
            elif 310 <= self.Y <= 360:
                self.X += self.v
                self.alpha = (1, 0)
            elif 240 <= self.X <= 290:
                self.Y += self.v
                self.alpha = (0, 1)
            elif 310 <= self.X <= 360:
                self.Y -= self.v
                self.alpha = (0, 1)

    def show(self):
        l = (int(self.X-self.alpha[0]*self.__length + self.alpha[1]*self.__width)-int(self.X+self.alpha[0]*self.__length - self.alpha[1]*self.__width))**2+\
            (int(self.Y-self.alpha[1]*self.__length - self.alpha[0]*self.__width)-int(self.Y+self.alpha[1]*self.__length + self.alpha[0]*self.__width))**2
        print(l)
        print(self.alpha)
        self.car = cv.create_polygon(int(self.X-self.alpha[0]*self.__length + self.alpha[1]*self.__width),
                                  int(self.Y-self.alpha[1]*self.__length - self.alpha[0]*self.__width),
                                  int(self.X-self.alpha[0]*self.__length - self.alpha[1] * self.__width),
                                  int(self.Y-self.alpha[1]*self.__length + self.alpha[0] * self.__width),
                                  int(self.X+self.alpha[0]*self.__length - self.alpha[1]*self.__width),
                                  int(self.Y+self.alpha[1]*self.__length + self.alpha[0]*self.__width),
                                  int(self.X + self.alpha[0] * self.__length + self.alpha[1] * self.__width),
                                  int(self.Y + self.alpha[1] * self.__length - self.alpha[0] * self.__width),
                                  fill='red')
        cv.update()


    def dt(self):
        cv.delete(self.car)

#road = 3.5/70
window = tk.Tk()
window.title("十字路口")
window.geometry("600x700")


cv = tk.Canvas(window, bg = 'black',height = 600,width = 600)
rect1 = cv.create_rectangle(230,0,370,600,fill = 'white')
rect2 = cv.create_rectangle(0,230,600,370,fill = 'white',outline = 'white')
line1 = cv.create_rectangle(300,0,300,600)
line2 = cv.create_rectangle(0,300,600,300)
cv.pack()


u = [CAR(2,1)]
def hit():
    t = 0
    while True:
        t += 1
        if t%30 == 0:
            x = [0,1,2,3]
            a = random.choice(x)
            x.remove(a)
            b = random.choice(x)
            u.append(CAR(a,b))
        for c in u:
            c.show()
        time.sleep(0.05)
        for c in u:
            c.dt()
            c.GO()



b = tk.Button(window,height = 650,width = 300,font = ("Arial",12), text="STRAT", command=hit)
b.pack()




window.mainloop()
