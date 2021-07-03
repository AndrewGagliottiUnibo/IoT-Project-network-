# -*- coding: utf-8 -*-
"""
Created on Sat Jul  3 20:07:42 2021

@author: andre
"""

import Devices.Device01 as dvc01
import Devices.Device02 as dvc02
import Devices.Device03 as dvc03
import Devices.Device04 as dvc04
import ServerAndGateway.gateway as SADg
import ServerAndGateway.server as SADs

dvc01.launch()
dvc02.launch()
dvc03.launch()
dvc04.launch()

SADg.launch()
SADs.launch()