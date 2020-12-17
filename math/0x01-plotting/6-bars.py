#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4,3))

names = ['Farrah', 'Fred', 'Felicia']
apples, bananas, oranges, peaches = fruit[0], fruit[1], fruit[2], fruit[3]
add1 = np.add(apples, bananas)
add2 = np.add(add1, oranges)
fig, axe = plt.subplots()

axe.bar(names, apples, 0.5, label='apples', color='red')
axe.bar(names, bananas, 0.5, bottom=apples, label='bananas', color='yellow')
axe.bar(names, oranges, 0.5, bottom=add1, label='oranges', color='#ff8000')
axe.bar(names, peaches, 0.5, bottom=add2, label='peaches', color='#ffe5b4')
axe.set_ylabel('Quantity of Fruit')
axe.set_ylim((0, 80))
axe.set_title('Number of Fruit per Person')
plt.legend()
plt.show()
