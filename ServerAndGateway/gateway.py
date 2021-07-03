# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 00:12:45 2021

@author: andre
"""

# Libraries used
import socket as sck
import time

# After the design of the basic fuctions and the creation of the single devices
# now it's time to create the gateway, which purpose is to firstly wait for the
# data collected by all devices and then eastablish a TCP connection to the cloud
#
# After establishing the connection now it's time to send the data and then waiting
# for the confirm of correct reception of the data.


def launch():
    
    # Variables
    nDevices = 4
    buffer = 4096
    measuresToSend = ''
    
    # First we start to listen for the devices: so we need the host ip address and 
    # the port number we are using
    sDevice = sck.socket(sck.AF_INET, sck.SOCK_DGRAM)
    sDevice.bind(("localhost", 10000))
    
    # Now it's time to wait for the detections - UDP connection
    for i in range(nDevices):
        data, address = sDevice.recvfrom(buffer)
        measuresToSend = measuresToSend + data.encode('utf-8') + '\n'
        time.sleep(2)
        print('Data receveived from {}' .format(address))
        sDevice.sendto('Data arrived.'.encode(), address)
        
    # Closing socket after receiving the data collected
    sDevice.close()
    
    """---------------------------------------------------------------------------------------------------------"""
    # Resetting the buffer
    buffer = 4096
    
    # Now it's time to send all collected data to cloud server and in order to do that 
    # we need to establish a TCP connection between the gateway and the server:
    # gateway is the sender
    print('... time to open interface 10.10.10.5')
    sCloud = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
    sCloud.connect(('localhost', 8002))
    start = time.time()
    
    # Sending data to server
    sCloud.send(measuresToSend.encode())
    
    # Now we have to wait for server response
    print('Waiting ...')
    data = sCloud.recv(buffer)
    
    #Data printed
    elapsed = time.time() - start    
    print('Message received: {}' .format(data.decode('utf-8')))
    print('Size of used buffer is {}' .format(buffer))
    print('Time occured for TCP establishing: {}' .format(elapsed))
    
    # CLosing
    sCloud.close()