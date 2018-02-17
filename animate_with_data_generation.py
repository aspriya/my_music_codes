import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
from random import randint

import numpy as np

style.use('seaborn-darkgrid')

fig, ax = plt.subplots()

x = []
y = []

def animate(i):
	x.append(i)
	y.append(randint(0,9))
	ax.clear()
	ax.plot(x, y)
	if(i==19.5):
		del y[:]
		del x[:]
		
ani = animation.FuncAnimation(fig, animate, frames=np.arange(0,20, 0.5), interval=50)
plt.show()