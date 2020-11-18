# File name : kinematic_processing.py
# Author : Simon Bastide
# Encoding : utf-8
# Function : Tools for kinematic data processing
################################################################################

import numpy as np
import matplotlib.pyplot as plt
import logging
from scipy import signal
from skspatial.objects import Points, Line

def movement_bounds(position, sample_rate, velocity = None, treshold = 5, relativ_treshold = True, plot = True):
    """Determine le début et la fin d'un mouvement. Le mouvement est considéré
     lorsque la vitesse du mouvement est supérieur au seuil (relatif ou absolu).
     Si le seuil est relatif (relativ_treshold = True), treshold doit être donné en % (5 par default)
     Si le seuil est absolue (relativ_treshold = False), treshold doit être donné dans la même unité que la vitesse (5 par default). 
     
     
     Notes : Il est préférable de donner en entré une position filtrée pour éviter que la vitesse
      calculée soit trop bruitée"""
    if velocity is None:
        velocity = np.diff(position)*sample_rate
        
    vel_abs = abs(velocity)
    if relativ_treshold:
        treshold = (treshold/100)*np.nanmax(vel_abs)
    max_vel_loc = np.nanargmax(vel_abs)
    starting_frame_find = False
    ending_frame_find = False
    for frame, vel_value in reversed(list(enumerate(vel_abs[:max_vel_loc]))):
        if (vel_value < treshold) and not starting_frame_find:
            start_frame = frame
            starting_frame_find = True
            break
                
    for frame, vel_value in (enumerate(vel_abs[max_vel_loc:])):
        if (vel_value < treshold) and not ending_frame_find:
            end_frame = max_vel_loc+frame
            ending_frame_find = True
            break

    if not starting_frame_find:
        start_frame = 1
        logging.warning("Début du mouvement non indentifié : fixé au début de l'enregistrement par défault")
    if not ending_frame_find:
        end_frame = len(velocity)
        logging.warning("Fin du mouvement non indentifié : fixé à la fin de l'enregistrement par défault")

    if plot:
        plt.figure()
        plt.title("Position")
        plt.plot(position, label = 'position')
        plt.axvline(x=start_frame, label = 'début mouvement', color = 'black')
        plt.axvline(x=end_frame, label = 'fin mouvement', color = 'brown')

        plt.legend()

        plt.figure()
        plt.title("Vitesse")
        plt.plot(abs(velocity), label = 'vitesse')
        #plt.plot(vel_abs, label = 'vitesse absolue')
        plt.axvline(x=max_vel_loc, label = 'Vitesse max', color = 'green')
        plt.axvline(x=start_frame, label = 'début mouvement', color = 'black')
        plt.axvline(x=end_frame, label = 'fin mouvement', color = 'brown')
        plt.axhline(y=treshold, label = 'seuil', color = 'red')
        plt.legend()
        plt.show()
    
    return (start_frame, end_frame)

def isupward(max_velocity, n_mov = None):
    if n_mov is None:
        n_mov = 1
    if max_velocity > 0 and n_mov % 2 == 1:
        return True
    elif max_velocity > 0 and n_mov % 2 == 0:
        #logging.warning("Mouvement n° {} à vérifier".format(n_mov))
        return True
    elif max_velocity < 0 and n_mov % 2 == 1:
        #logging.warning("Mouvement n° {} à vérifier".format(n_mov))
        return False
    else:
        return False

def get_id_dict(subj_id = None, cond_id = None, block_id = None, mov_id = None):

     return {'subject':subj_id, 'condition': cond_id, 'block': block_id,\
         'movement': mov_id}

def get_position_params(position, sample_rate, extension = ""):

    duration = len(position)/sample_rate
    amplitude = position[-1]-position[0]

    return {'duration' + extension : duration, 'amplitude' + extension : amplitude}

def get_velocity_params(velocity, sample_rate, extension = ""):

    duration = len(velocity)/sample_rate
    mean_vel = np.mean(abs(velocity))
    max_vel_loc = np.argmax(abs(velocity))
    max_vel = velocity[max_vel_loc]
    time_to_peak_vel = max_vel_loc * 1/sample_rate
    

    return {'duration' + extension : duration, 'max_vel' + extension : max_vel, 'time_to_peak_vel' + extension : time_to_peak_vel,\
         'mean_vel' + extension : mean_vel}

def get_max_vel_loc(velocity):
    return np.argmax(abs(velocity))
    
def get_acceleration_params(acceleration, max_vel_loc, sample_rate, extension = ""):
    
    max_accel_loc = np.argmax(abs(acceleration[:max_vel_loc]))
    max_decel_loc = max_vel_loc + np.argmax(abs(acceleration[max_vel_loc:]))
    max_accel = acceleration[max_accel_loc]
    max_decel = acceleration[max_decel_loc]
    mean_accel = np.mean(acceleration[:max_vel_loc])
    mean_decel = np.mean(acceleration[max_vel_loc:])
    time_to_peak_acc = max_accel_loc * 1/sample_rate

    return {'max_accel' + extension : max_accel, 'time_to_peak_acc' + extension : time_to_peak_acc,\
         'mean_accel' + extension : mean_accel, 'max_decel' + extension : max_decel,\
             'mean_decel' + extension : mean_decel}



def params(position, velocity = None, acceleration = None, sample_rate = 100, plot = True, suffix = ""):
    """Calcul des paramètres simples sur le mouvement. La position en entrée doit étre la
    position du mouvement effectif (utiliser movement_bounds avant pour selectionner le mouvement effectif).
    Il est préférable de donner la position filtrée en entrée. 
    Cette fonction retourne un dictionnaire conteannt les paramétres calculés."""

    if velocity is None:
        velocity = np.diff(position)*sample_rate
    if acceleration is None:
        acceleration = np.diff(velocity)*sample_rate


    try:
        duration = len(position)/sample_rate
        amplitude = position[-1]-position[0]
        mean_vel = np.mean(abs(velocity))
        peak_vel_loc = np.argmax(abs(velocity))
        peak_accel_loc = np.argmax(abs(acceleration[:peak_vel_loc]))
        peak_decel_loc = peak_vel_loc + np.argmax(abs(acceleration[peak_vel_loc:]))
        peak_vel = velocity[peak_vel_loc]
        peak_accel = acceleration[peak_accel_loc]
        peak_decel = acceleration[peak_decel_loc]
        mean_accel = np.mean(acceleration[:peak_vel_loc])
        mean_decel = np.mean(acceleration[peak_vel_loc:])

        time_to_peak_vel = peak_vel_loc * 1/sample_rate
        time_to_peak_acc = peak_accel_loc * 1/sample_rate
        time_to_peak_decel = peak_decel_loc * 1/sample_rate


        if plot:
            time = np.linspace(0, duration, len(position))
            plt.figure
            plt.subplot(3, 1, 1)
            plt.title("Position")
            plt.plot(time, position)
            plt.subplot(3, 1, 2)
            plt.title("Vitesse")
            plt.plot(time, velocity)
            plt.scatter(time[peak_vel_loc], velocity[peak_vel_loc], label = 'Vitesse max', color = 'green', marker = '*')
            plt.legend()
            plt.subplot(3, 1, 3)
            plt.title("Acceleration")
            plt.plot(time, acceleration)
            plt.scatter(time[peak_accel_loc], acceleration[peak_accel_loc], label = 'Acceleration max', color = 'green', marker = '*')
            plt.scatter(time[peak_decel_loc], acceleration[peak_decel_loc], label = 'Deceleration max', color = 'red', marker = '*')
            plt.xlabel("Time (s)")
            plt.legend()

            plt.tight_layout()
            plt.show()
    
    except ValueError:
        logging.warning("Unable to compute parameter")
        duration = None
        amplitude = None
        mean_vel = None
        peak_vel = None
        peak_accel = None
        peak_decel = None
        mean_accel = None
        mean_decel = None
        time_to_peak_vel = None
        time_to_peak_acc = None
        time_to_peak_decel = None

    return ({
        'MD' + suffix : duration,
        'A' + suffix : amplitude,
        'PV' + suffix : peak_vel,
        'tPV' + suffix : time_to_peak_vel,
        'mV' + suffix : mean_vel,
        'PA' + suffix : peak_accel, 
        'tPA' + suffix : time_to_peak_acc,
        'mA'+ suffix :  mean_accel, 
        'PD'+ suffix : peak_decel,
        'tPD' + suffix : time_to_peak_decel, 
        'mD' + suffix : mean_decel})

def unit_vector(vector):
    """ Returns the unit vector of a vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1_array, v2_array):
    """ Returns an array of angle in radians between vectors 'v1_array' and 'v2_array' """
    angle = list()
    if len(v1_array) == len(v2_array):
        for v1,v2 in zip(v1_array,v2_array):
            v1_u = unit_vector(v1)
            v2_u = unit_vector(v2)
            angle.append(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))
    else:
        print("Arrays must have the same length ")
    return np.array(angle)

def tangential_velocity(pos, sample_rate):
    """Retroune la vitesse tangentielle du signal """

    if pos.shape[0] < pos.shape[1]:
        pos = pos.transpose()
    vel = np.diff(pos,axis = 0)
    vel = np.sqrt(np.sum(np.square(vel), axis = 1))*sample_rate
    return np.append(vel, vel[-1])

def travelled_distance(vel, sample_rate):
    """Retourne la distance parcourue en fonction du temps"""
    return np.concatenate(([0], np.cumsum(vel)))/sample_rate


def fit_line_direction(points):
    direction = list()
    for frame in range(0,len(points)):
        points_ = Points(np.reshape(np.array(points[frame,:]), (-1,3)))
        line_fit = Line.best_fit(points_)
        direction.append(np.array(line_fit.direction))
    return np.array(direction) 