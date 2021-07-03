# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 22:41:11 2021

@author: andre
"""

import SupportFunctions as sf

ip = '192.168.1.1'
fileName = 'Measures01.txt'
gateway_address = ('localhost', 10000)
buffer = 4096
measures = sf.detectionsReader(ip, fileName)
sf.gatewayConnection(gateway_address, measures, buffer)