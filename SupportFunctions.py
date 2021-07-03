# -*- coding: utf-8 -*-
"""
Created on Fri Jul  2 23:04:10 2021

@author: andre
"""

# Libraries used
import time
import socket as sck

# In this first istance I decided to create some usefull functions for supporting
# my application and giving my code a nice look.
#
# I start by defining the real core of the application, which means reading the detections
# coming from the devices and finding a way to establish a UDP connection with
# the Gateway

# I start defining the way the application reads the detections:
# first we start opening the file, then reading each line of it and at the end
# the function saves all data in a field that must be returned, which are our measurations.
def detectionsReader(ip, fileName):
    
    #Opening the file
    path = 'Data/' + fileName
    file = open(path, 'r')
    print('Reading available data ... ')
    time.sleep(2)
    
    #Reading all measures and creating a list of strings
    measures = ip + file.readlines()

    # Closing the file and returning the list of strings
    file.close()
    return measures

"""---------------------------------------------------------------------------------------------------------"""
# Now it's time to create the UDP connection and sending the collected data to
# the Gateway: in order to do this I need the Gateway address and the measures
# that have been collected from a specific device
def gatewayConnection(address, measures, buffer):
    
    # Establishing UDP connection - starting by creating the socket
    # and then sending info using a try statement for controlling any kind of
    # exceptions
    print('Opening socket ...')
    mySocket = sck.socket(sck.AF_INET, sck.SOCK_DGRAM)
    
    try:
        # Sending data and then ...
        time.sleep(2)
        print('Sending ...')
        mySocket.sendto(measures.encode(), address)
        
        # recvfrom() reads a number of bytes sent from an UDP socket, in this case
        # we are reading data from the Gateway - it's like waiting for a response
        print('Waiting ...')
        data, server = mySocket.recvfrom()
        time.sleep(2)
        print('Message: {}' .format(data.decode("utf8")))
        print('Size of used buffer is {}' .format(buffer))
        
    except Exception as e:
        print(e)
    finally:
        print('Closing socket ...')
        mySocket.close()
        
"""---------------------------------------------------------------------------------------------------------"""
# Now I'm defining the last part of the project: after defining the devices and
# the Gateway, now I'm defining TCP connection to the Gateway established by the server
#
# It'a Simple TCP connection with a data printing at the end
def connectionToGateway(serverPort, serverIP, buffer):
    
    print('Establishing TCP connection ... \n')
    sSocket = sck.socket(sck.AF_INET, sck.SOCK_STREAM)
    sSocket.bind(('localhost', serverPort))
    
    # Listening the request for connection
    print('Interface: {}, port: {}' .format(serverIP, serverPort))
    sSocket.listen(1)
    
    # Accepting the connection and then it's time to receive data
    gatewayConnection, address = sSocket.accept()
    print("Gateway connected! \n")
    print("Detections received are:\n")
    serverMessage = gatewayConnection.recv(buffer)
    
    # Printing data
    print(serverMessage.decode("utf8"))
    gatewayConnection.send(("Data received").encode())
    
    # Closing
    gatewayConnection.close()
    sSocket.close()
    
    
    
    
    
    