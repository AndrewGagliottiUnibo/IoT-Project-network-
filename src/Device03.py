# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:01:16 2021

@author: andre
"""

import Functions.SupportFunctions as sf

# We are defining ip address, name of the related file, gateway address to 
# connect with.
# After this we start reading the file content and send it to the gateway
ip = '192.168.1.3'
fileName = 'Measures03.txt'
portNumberUDP = 10024
gatewayAddress = ('localhost', portNumberUDP)
buffer = 4096

measures = sf.detectionsReader(ip, fileName)
sf.gatewayConnection(gatewayAddress, measures, buffer)