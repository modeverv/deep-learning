#! /usr/bin/env python
# -*- coding: utf-8 -*-

hungry = True
if hungry:
    print("おなかがすいた")
else:
    print("おなかがすいていない")

for i in [1,2,3]:
    print(i)


def hello():
    print("hello")

hello()

class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("GoodBy " + self.name +"!")

m = Man("David")
m.hello()
m.goodbye()


import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
 
x = np.arange(0,6,0.1)
y = np.sin(x)
plt.plot(x,y)
#plt.show()
plt.savefig("/var/www/plot/1f.png")

