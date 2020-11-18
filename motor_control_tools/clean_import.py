"""This module contains tools for cleaning our datas"""

# author : Simon Bastide
# mail : simon.bastide@outlook.com

import os
import pandas as pd
import numpy as np


def robot_data(subject_label, condition_label, block_number, raw_data_path):
    """Function to aggregate the datas of the robot. Only datas of the 4th axis are conserved.

    Args:
        subject_label (str): label of the subject
        condition_label (str): label of the condition
        block_number (int): number of the bloc
        raw_data_path (str): path to the file were are contained the raw datas

    Returns:
        pandas.Dataframe: One data frame containing the datas
    """
    missing_data = []
    path =  os.path.join(raw_data_path,subject_label)
    if condition_label in ['SE', 'SE_out', 'SA_out', 'SA_in']:
        print("No robot data for SE condition")
        return False        
    else:
        suffix = subject_label + '_' + condition_label + '-' + str(block_number) + '.txt'

        # iteration times
        file_path = os.path.join(path,"iteration_times_" + suffix)
        if os.path.isfile(file_path):
            data = np.genfromtxt(file_path, delimiter = ';')
            if np.isnan(data[-1]):
                data = data[:-1]
            df = pd.DataFrame(data, columns = ['iteration_time'])
        else:
            missing_data.append(True)

        # positions
        file_path = os.path.join(path,"exo_positions_" + suffix)
        if os.path.isfile(file_path):
            data = np.genfromtxt(file_path, delimiter = ' ; ')
            data = np.reshape(data, (-1, 4))
            df = df.assign(position = data[:len(df),3])
        else:
            missing_data.append(True)


        # vitesses
        file_path = os.path.join(path, "exo_vitesses_" + suffix)
        if os.path.isfile(file_path):
            data = np.genfromtxt(file_path, delimiter = ' ; ')
            data = np.reshape(data, (-1, 4))
            df = df.assign(velocity = data[:len(df),3])
        else:
            missing_data.append(True)

        # Courants
        file_path = os.path.join(path, "exo_courants_" + suffix)
        if os.path.isfile(file_path):
            data = np.genfromtxt(file_path, delimiter = ' ; ')
            data = np.reshape(data, (-1, 4))
            df = df.assign(courant = data[:len(df),3])
        else:
            missing_data.append(True)

        # Force Sensor
        file_path = os.path.join(path, "exo_courants_" + suffix)
        if os.path.isfile(file_path):
            data = np.genfromtxt(os.path.join(path, "F_T_Sensor_measures_" + suffix), delimiter = ';')
            if np.isnan(data[-1]):
                data = data[:-1]
            data = np.reshape(data[:len(df)*6], (-1, 6))
            df = df.assign(
                Fx = data[:len(df),0],
                Fy = data[:len(df),1],
                Fz = data[:len(df),2],
                )
        else:
            missing_data.append(True)
        
        if any(missing_data):
            print("!! No robot data !!")
            return False
        else:
            return df

def emg_data(subject_label, condition_label, block_number, raw_data_path):
    """[summary]

    Args:
        subject_label ([type]): [description]
        condition_label ([type]): [description]
        block_number ([type]): [description]
        raw_data_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    path =  os.path.join(raw_data_path, subject_label)
    if condition_label == 'SE' or condition_label == 'SE_out':
        file_name = subject_label + '_' + condition_label + '_a.tsv'
    else:
        file_name = subject_label + '_' + condition_label + '-' + str(block_number) + '_a.tsv'
    
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep = "\t", engine = 'python', skiprows=13)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.drop('SAMPLE', axis = 1)
        df = df.rename({'TIME':'time'}, axis = 1)
    else:
        print("No emg data")
        return False
        
    return df

def kin_data(subject_label, condition_label, block_number, raw_data_path):
    """[summary]

    Args:
        subject_label ([type]): [description]
        condition_label ([type]): [description]
        block_number ([type]): [description]
        raw_data_path ([type]): [description]

    Returns:
        [type]: [description]
    """

    path =  os.path.join(raw_data_path, subject_label)
    if condition_label in ['SE', 'SE_out', 'SA_out', 'SA_in']:
        file_name = subject_label + '_' + condition_label + '.tsv'
    else:
        file_name = subject_label + '_' + condition_label + '-' + str(block_number) + '.tsv'
    
    file_path = os.path.join(path, file_name)
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, sep = "\t", engine = 'python', skiprows=10)
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        df = df.drop('Frame', axis = 1)
        df = df.rename({'Time':'time'}, axis = 1)
    else:
        print("No kinematic data")
        return False

    return df


def get_mass_and_deltaq(data_path, subj_id):
    """Renvoi la masse de l'avant bras du sujet 
    et le décalage angulaire moyen identifié
    au cours de la manip entre l'avant bras de sujet et
    l'axe 4 de l'exo

    Args:
        subj_id ([type]): [description]
    """
    file_name = os.path.join(
        data_path,
        subj_id,
        "human_limb_identification_" + subj_id + ".txt",
    )
    with open(file_name, 'r') as f:
        values = f.read()
    return float(values.split(" ")[0]), float(values.split(" ")[1])


def get_anthropo(data_path, subj_id):
    file_name = os.path.join(data_path, "infos_participants.xlsx")
    forearm_mass_estim, delta_q_estim = get_mass_and_deltaq("raw_data", subj_id)
    data = pd.read_excel(file_name).query("subject == @subj_id").to_dict('records')[0]
    data.update({
        "forearm mass estimation" : forearm_mass_estim,
        "angular offset estimation" : delta_q_estim
    })
    return data
