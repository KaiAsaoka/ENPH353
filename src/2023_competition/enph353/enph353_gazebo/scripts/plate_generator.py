#!/usr/bin/env python3

import cv2
import csv
import numpy as np
import os
import pyqrcode
import random
import string

from random import randint
from PIL import Image, ImageFont, ImageDraw

entries = {'SIZE': ["TWO", "314", "DOZEN", "RAYO10", "COUNTLESS", "LEGIONS",
                    "TRIPLETS"],
           'VICTIM': ["PARROTS", "ROBOTS", "BACTERIA", "JEDIS", "ALIENS", 
                      "CITIZENS", "PHYSICISTS", "FRODO", "DINOSAURS", "BUNNIES",
                      "BED BUGS", "ANTS"],
           'CRIME': ["STEAL", "TRESPASS", "LIE TO", "DESTROY", "PUNCH", "BITE", 
                     "TRANSMOGRIFY", "TELEPORT", "ACCELERATE", "IRRADIATE",
                     "CURSE", "HEADBUT", "DEFRAUD", "DECELERATE", "TICKLE"],
           'TIME': ["NOON", "MIDNIGHT", "DAWN", "DUSK", "JURASIC", "TWILIGHT",
                    "D DAY", "Q DAY", "2023", "WINTER", "SUMMER", "SPRING",
                    "AUTUMN"],
           'PLACE': ["HOSPITAL", "MALL", "FOREST", "MOON", "CLASS", "BEACH", 
                     "JUNGLE", "BASEMENT", "THE HOOD", "SEWERS", "CAVE",
                     "BENU", "MARS"],
           'MOTIVE': ["GLUTTONY", "CURIOSITY", "IGNORANCE", "FEAR", "PRIDE", 
                      "LOVE", "REVENGE", "PASSION", "BOREDOM", "THRILL", 
                      "GREED", "FAME", "ACCIDENT", "HATE", "SELF DEFENSE"],
           'WEAPON': ["STICK", "ROCKET", "ANTIMATTER", "NEUTRINOS", "SHURIKEN", 
                      "PENCIL", "PLASMA", "WATER", "FIRE", "POTATO GUN", 
                      "ROPE", "ELECTRON", "HIGH VOLTAGE", "POLONIUM"],
           'BANDIT': ["EINSTEIN", "PIKACHU", "SHREK", "LUIGI", "BARBIE", 
                      "BATMAN", "CAESAR", "SAURON", "THANOS", "GOKU", 
                      "CAO CAO", "THE DEVIL", "GODZILA", "TEMUJIN", 
                      "HANNIBAL"]
           }

# Find the path to this script
SCRIPT_PATH = os.path.dirname(os.path.realpath(__file__)) + "/"
TEXTURE_PATH = '../media/materials/textures/'

banner_canvas = cv2.imread(SCRIPT_PATH+'clue_banner.png')
PLATE_HEIGHT = 600
PLATE_WIDTH = banner_canvas.shape[1]
IMG_DEPTH = 3

# write plates to plates.csv
with open(SCRIPT_PATH + "plates.csv", 'w') as plates_file:
    csvwriter = csv.writer(plates_file)

    i = 0
    for key in entries:
        # pick a random criminal
        j = random.randint(0, len(entries[key])-1)
        random_value = entries[key][j]

        entry = key + "," + random_value
        print(entry)
        csvwriter.writerow([key, random_value])

        # Generate plate
   
        # To use monospaced font for the license plate we need to use the PIL
        # package.
        # Convert into a PIL image (this is so we can use the monospaced fonts)
        blank_plate_pil = Image.fromarray(banner_canvas)
        # Get a drawing context
        draw = ImageDraw.Draw(blank_plate_pil)
        font_size = 90
        monospace = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf", 
                                       font_size)
        font_color = (255,0,0)
        draw.text((250, 30), key, font_color, font=monospace)
        draw.text((30, 250), random_value, font_color, font=monospace)
        # Convert back to OpenCV image and save
        populated_banner = np.array(blank_plate_pil)

        # Save image
        cv2.imwrite(os.path.join(SCRIPT_PATH+TEXTURE_PATH+"unlabelled/",
                                 "plate_" + str(i) + ".png"), populated_banner)
        i += 1