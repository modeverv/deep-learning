# conding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# データの作成
x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みで生成
y = np.sin(x)

# グラフの描画
#plt.plot(x, y)
#plt.show()

# リスト
a = [1,2,3,4,5]
print(a[2])
print(a[0:2])

# ディクショナリ
me = {'height': 100}
print(me['height'])
me['weight'] = 70
print(me)

# ブーリアン
hungry = True
sleepy = False

# if !とか使えないのね notねあとelif
if not hungry:
    print("I'm hungry")
elif not sleepy:
    print("I'm sleepy")
else:
    print("I'm not hungry")

# for
for i in [1,2,3]:
    print(i)

# 関数
def hello():
    print("Hello World")

hello()

class Man:
    def __init__(self,name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print("Hello " + self.name + "!")

    def goodbye(self):
        print("Good bye " + self.name + "!")

m = Man("nora inu")
m.hello()
m.goodbye()



