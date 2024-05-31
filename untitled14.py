# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:51:56 2024

@author: MAT-Admin
"""

# importing all required libraries
import telebot
from telethon.sync import TelegramClient
from telethon.tl.types import InputPeerUser, InputPeerChannel
from telethon import TelegramClient, sync, events


# get your api_id, api_hash, token
# from telegram as described above
api_id = '14708554'
api_hash = '257db95d60f0c0aac4b2327906846d21'
token = '1357690578:AAHjhI2ctKe42EPlAXWeQ3rMBGrpA-lyEmU'
message = "Working..."

# your phone number
phone = '+6281802414034'

# creating a telegram session and assigning
# it to a variable client
client = TelegramClient('session', api_id, api_hash)

# connecting and building the session
client.connect()

# in case of script ran first time it will
# ask either to input token or otp sent to
# number or sent or your telegram id 
if not client.is_user_authorized():

	client.send_code_request(phone)
	
	# signing in the client
	client.sign_in(phone, input('Enter the code: '))


try:
	# receiver user_id and access_hash, use
	# my user_id and access_hash for reference
	receiver = InputPeerUser('user_id', 'user_hash')

	# sending message using telegram client
	client.send_message(receiver, message, parse_mode='html')
except Exception as e:
	
	# there may be many error coming in while like peer
	# error, wrong access_hash, flood_error, etc
	print(e);

# disconnecting the telegram session 
client.disconnect()
