# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 17:44:46 2021

@author: andre
"""

import random

# Change here if you want more devices: remember to create new devices!
timesToRepeat = 4

# A dummy file content generator, it literally has a for loop with write method
for j in range(timesToRepeat):
    file = open('../../../Data/Measures0{}.txt' .format(j + 1), 'w')
    for i in range(timesToRepeat):
        file.write('Time: {}:00 - ' .format(00 + i * timesToRepeat * 2) + 'Temperature: {} C -' .format(round(random.uniform(20, 40), 1)) + ' {} % of humidity' .format(round(random.uniform(40, 80), 1)) + '\n')
    
    file.close()