import pygame as pg
import numpy as np
import sys
import random
import time

# Create the environment

class Bar_Builder():
    def __init__(self, max_dist_y, n_episode, n_run):
        pg.init()
        self.screen_width = 600
        self.screen_height = 800
        # fonts=pg.font.get_fonts()
        # font_style=random.choice(fonts)
        # print(font_style)
        self.font = pg.font.SysFont('stencil', 20)
        self.screen = pg.display.set_mode((self.screen_width, self.screen_height))
        self.SCALE = self.screen_height*0.9/max_dist_y
        self.goal_dist = max_dist_y
        self.n_episode = n_episode
        self.n_run = n_run

    def display_bar(self, position, radius, height):
        # Display the bar
        pg.draw.rect(self.screen, (169,169,169), (self.screen_width/2-radius*self.SCALE, position*self.SCALE-height*self.SCALE, radius*2*self.SCALE, height*self.SCALE))
        pg.display.update()
        pg.time.delay(50)
    
    def display_goal(self, goal_dist):
        # Display the goal
        pg.draw.rect(self.screen, (0, 0, 255), (self.screen_width / 2 - 5 , goal_dist * self.SCALE - 5, 10, 10))
        pg.display.update()
        pg.time.delay(50)

    def display_total_mass(self, total_mass):
        # Display the score

        text = self.font.render('Total mass: ' + str(total_mass)+ '[kg]', True, (255, 255, 0))
        self.screen.blit(text, (3, 5))
        pg.display.update()

    def display_total_deformation(self, total_deformation):
        # Display the score 
        text = self.font.render('Total deformation: ' + str(total_deformation) + '[µm]', True, (255, 255, 0))
        self.screen.blit(text, (3, 25))
        pg.display.update()

    def goal_force_vector(self, goal_dist):
        # draw a red arrow pointing to the goal
        pg.draw.line(self.screen, (255, 0, 0), (self.screen_width/2, goal_dist*self.SCALE), (self.screen_width/2, 1.05*goal_dist*self.SCALE), 5)
        pg.draw.polygon(self.screen, (255, 0, 0), ((self.screen_width/2-5, 1.05*goal_dist*self.SCALE), (self.screen_width/2+5, 1.05*goal_dist*self.SCALE), (self.screen_width/2, 1.07*goal_dist*self.SCALE)))
        pg.display.update()

    def close(self):
        pg.time.delay(2000)
        pg.quit()

    def run_render(self, list_render):
        # Run the render
        for i in range(len(list_render)):
            # Display the bar
            self.display_bar(list_render[i,0], list_render[i,4], list_render[i,5])
        self.display_goal(self.goal_dist)
        self.goal_force_vector(self.goal_dist)
        self.display_total_mass(np.round(list_render[-1][1],5))
        self.display_total_deformation(np.round(list_render[-1][2],5))
        pg.image.save(self.screen, './out/renders/'+self.n_run+'_'+'render'+str(self.n_episode)+'.png')
        self.close()
