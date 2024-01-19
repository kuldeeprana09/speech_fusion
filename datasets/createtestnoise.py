#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 17 22:11:19 2021

@author: lab70809
"""

import pandas as pd


noise = pd.read_csv('noisecvtt.csv', header = None)
noise.columns = ['name', 'path']
print(noise.head())
music = noise[noise['name'] == 'music']
Music = noise[noise['name'] == 'Music']
music = pd.concat([music, Music], ignore_index=True)
bus = noise[noise['name'] == 'BUS']
BUS = noise[noise['name'] == 'Bus']
bus = pd.concat([bus, BUS], ignore_index=True)
cafe = noise[noise['name'] == 'CAF']
ped = noise[noise['name'] == 'PED']
street = noise[noise['name'] == 'STR']
door = noise[noise['name'] == 'door']
bird = noise[noise['name'] == 'Bird']
fan = noise[noise['name'] == 'fan']
fowl = noise[noise['name'] == 'Fowl']

music.to_csv('musicNoise.csv', index=False, header = False)
bus.to_csv('busNoise.csv', index = False, header = False)
street.to_csv('STRNoise.csv', index = False, header = False)
cafe.to_csv('cafeNoise.csv', index = False, header = False)
ped.to_csv('PEDNoise.csv', index = False, header = False)
door.to_csv('doorNoise.csv', index = False, header = False)
bird.to_csv('birdNoise.csv', index = False, header = False)
fan.to_csv('fanNoise.csv', index = False, header = False)
fowl.to_csv('fowlNoise.csv', index = False, header = False)