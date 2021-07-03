# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:56:33 2021

@author: andre
"""

# Libraries used
import Functions.SupportFunctions as sf

serverPort = 8002
serverIP = '10.10.10.5'
buffer = 4096

# Establishing the connection
sf.connectionToGateway(serverPort, serverIP, buffer)
