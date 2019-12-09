# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 10:46:06 2019

@author: Vladimir
"""

import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

global x_food, food_lo, food_md, food_hi
global x_serv, serv_lo, serv_md, serv_hi
global y_tip, tip_lo, tip_md, tip_hi

x_food = np.arange(0, 11, 1)
x_serv = np.arange(0, 11, 1)
y_tip = np.arange(0, 26, 1)

food_lo = fuzz.trimf(x_food, [0, 0, 5])
food_md = fuzz.trimf(x_food, [0, 5, 10])
food_hi = fuzz.trimf(x_food, [5, 10, 10])
serv_lo = fuzz.trimf(x_serv, [0, 0, 5])
serv_md = fuzz.trimf(x_serv, [0, 5, 10])
serv_hi = fuzz.trimf(x_serv, [5, 10, 10])
tip_lo = fuzz.trimf(y_tip, [0, 0, 13])
tip_md = fuzz.trimf(y_tip, [0, 13, 25])
tip_hi = fuzz.trimf(y_tip, [13, 25, 25])

fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

ax0.plot(x_food, food_lo, 'b', linewidth=1.5, label='Bad')
ax0.plot(x_food, food_md, 'g', linewidth=1.5, label='Decent')
ax0.plot(x_food, food_hi, 'r', linewidth=1.5, label='Great')
ax0.set_title('Качество пищи')
ax0.legend()

ax1.plot(x_serv, serv_lo, 'b', linewidth=1.5, label='Poor')
ax1.plot(x_serv, serv_md, 'g', linewidth=1.5, label='Acceptable')
ax1.plot(x_serv, serv_hi, 'r', linewidth=1.5, label='Amazing')
ax1.set_title('Качество сервиса')
ax1.legend()

ax2.plot(y_tip, tip_lo, 'b', linewidth=1.5, label='Low')
ax2.plot(y_tip, tip_md, 'g', linewidth=1.5, label='Medium')
ax2.plot(y_tip, tip_hi, 'r', linewidth=1.5, label='Hidh')
ax2.set_title('Сумма чаевых')
ax2.legend()

def FuzzyInfluence(x_food_d, x_serv_d):
    food_level_lo = fuzz.interp_membership(x_food, food_lo, x_food_d)
    food_level_hi - fuzz.interp_membership(x_food, food_hi, x_food_d)
    food_level_lo = fuzz.interp_membership(x_serv, serv_lo, x_serv_d)