# -*- coding: utf-8 -*-
"""
Created on Thu Jun 27 15:52:14 2024

@author: MAT-Admin
"""
import os
from configparser import ConfigParser
config = ConfigParser()
isExistINI = os.path.exists('frame.ini')
if isExistINI:
    print("exist")
    f = open("frame.ini", "r")
    status = f.read()
    print("status: " + status)
    f.close()
    if status:
        try:
            os.system(f' python cctv_capture_raspi4_juni.py')
        except FileNotFoundError:
            print(f"Error: The file 'hello.py' does not exist.")
else:
    print("not exist")
    f = open("frame.ini", "x")
    f.write("0")
    f.close()