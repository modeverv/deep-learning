# conding: utf-8

import numpy as np
import matplotlib.pyplot as plt

# データの作成
#x = np.arange(0, 6, 0.1) # 0から6まで0.1刻みで生成
#y = np.sin(x)

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

# クラス
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

# NumPy
x = np.array([1.0, 2.0, 3.0])
print(x)
print(type(x))
y = np.array([2.0, 4.0, 6.0])
z = x + y
print(z)
z = x * y
print(z)
z = x / y
print(z)

A = np.array([[1, 2], [3, 4]])
print(A)
print(A.shape)
B = np.array([[3,0], [0,6]])

print(A + B)
print(A * B)

X = np.array([[51,55],[14,19],[0,4]])

print(X[X>15])

x = np.arange(0,6,0.1)
y = np.sin(x)
#plt.plot(x,y)
#plt.show()

y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x,y1,label ="sin")
plt.plot(x,y2,linestyle="--",label="cos")
plt.xlabel("x")
plt.ylabel("y")
plt.title("sin & cos")
plt.legend()
plt.show()








