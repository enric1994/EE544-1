#!/usr/bin/env python
# -*- coding: utf-8 -*-
# (c) Copyright 2019 Enric Moreu. All Rights Reserved.

import telebot
import os

telegramToken = os.environ.get('TELEGRAM_TOKEN')
telegramChatID = os.environ.get('TELEGRAM_ID')

def send(message):
	bot = telebot.TeleBot(telegramToken)
	bot.config['api_key'] = telegramToken
	bot.send_message(int(telegramChatID), message)
	print(message)