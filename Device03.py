# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:01:16 2021

@author: andre
"""


import SupportFunctions as sf

ip = '192.168.1.3'
fileName = 'Measures03.txt'
gateway_address = ('localhost', 10000)
measures = sf.detectionsReader(ip, fileName)
sf.gatewayConnection(gateway_address, measures)