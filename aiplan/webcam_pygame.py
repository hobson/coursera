#!/usr/bin/env python

import sys
import os
import pygame
import pygame.camera
from pygame.locals import *

SCREEN_SIZE = (int(640*.9), int(480*.9))

pygame.init()
pygame.camera.init()

camera_list = pygame.camera.list_cameras()
if not camera_list:
    print 'Tidak ada kamera terdeteksi!.'
    sys.exit(0)

screen = pygame.display.set_mode(SCREEN_SIZE)
pygame.display.set_caption(os.path.basename(__file__))


camera = pygame.camera.Camera(camera_list[0], SCREEN_SIZE)
camera.start()

while True:

    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)

    image = camera.get_image()
    screen.blit(image, (0,0))
    pygame.display.update()
