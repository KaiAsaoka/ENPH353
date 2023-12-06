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

entries = {'SIZE': ["100", "10 GOOGLES", "314", "A PAIR", "BAKER DOZEN",
                    "COUNTLESS", "DOZEN", "FEW", "FIVE", "HALF DOZEN",
                    "LEGIONS", "MANY", "QUINTUPLETS", "RAYO10", "SINGLE",
                    "THREE", "TRIPLETS", "TWO", "UNCOUNTABLE", "ZEPTILLION"],
           'VICTIM': ["ALIENS", "ANTS", "BACTERIA", "BED BUGS", "BUNNIES",
                      "CITIZENS", "DINOSAURS", "FRODOS", "JEDIS", "KANGAROO",
                      "KOALAS", "PANDAS", "PARROTS", "PHYSICISTS", "QUOKKAS",
                      "ROBOTS", "RABBITS", "TOURISTS", "ZOMBIES"],
           'CRIME': ["ACCELERATE", "BITE", "CURSE", "DECELERATE", "DEFRAUD",
                     "DESTROY", "HEADBUT", "IRRADIATE", "LIE TO", "POKE",
                     "PUNCH", "PUSH", "SCARE", "STEAL", "STRIKE", "SWEAR",
                     "TELEPORT", "THINKING", "TICKLE", "TRANSMOGRIFY",
                     "TRESPASS"],
           'TIME': ["2023", "AUTUMN", "DAWN", "D DAY", "DUSK", "EONS AGO",
                    "JURASIC", "MIDNIGHT", "NOON", "Q DAY", "SPRING",
                    "SUMMER", "TOMORROW", "TWILIGHT", "WINTER", "YESTERDAY"],
           'PLACE': ["AMAZON", "ARCTIC", "BASEMENT", "BEACH", "BENU", "CAVE",
                     "CLASS", "EVEREST", "EXIT 8", "FIELD", "FOREST",
                     "HOSPITAL", "HOTEL", "JUNGLE", "MADAGASCAR", "MALL",
                     "MARS", "MINE", "MOON", "SEWERS", "SWITZERLAND",
                     "THE HOOD", "UNDERGROUND", "VILLAGE"],
           'MOTIVE': ["ACCIDENT", "BOREDOM", "CURIOSITY", "FAME", "FEAR",
                      "FOOLISHNESS", "GLAMOUR", "GLUTTONY", "GREED", "HATE",
                      "HASTE", "IGNORANCE", "IMPULSE", "LOVE", "LOATHING",
                      "PASSION", "PRIDE", "RAGE", "REVENGE", "REVOLT",
                      "SELF DEFENSE", "THRILL", "ZEALOUSNESS"],
           'WEAPON': ["ANTIMATTER", "BALOON", "CHEESE", "ELECTRON", "FIRE",
                      "FLASHLIGHT", "HIGH VOLTAGE", "HOLY GRENADE", "ICYCLE",
                      "KRYPTONITE", "NEUTRINOS", "PENCIL", "PLASMA",
                      "POLONIUM", "POSITRON", "POTATO GUN", "ROCKET", "ROPE",
                      "SHURIKEN", "SPONGE", "STICK", "TAMAGOCHI", "WATER",
                      "WRENCH"],
           'BANDIT': ["BARBIE", "BATMAN", "CAESAR", "CAO CAO", "EINSTEIN",
                      "GODZILA", "GOKU", "HANNIBAL", "L", "LENIN", "LUCIFER",
                      "LUIGI", "PIKACHU", "SATOSHI", "SHREK", "SAURON",
                      "THANOS", "TEMUJIN", "THE DEVIL", "ZELOS"]
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

        #if len(random_value) < 11:
            #random_value = random.choice(string.ascii_uppercase) + " " + random_value

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