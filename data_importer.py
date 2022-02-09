'''
This module contains a tool kit for loading data in the app_model_validation.py file
'''

import os
import pandas as pd

training_data_path = 'data/training/220128.csv'

def load_train_and_validation_data(frac_ = 0.8):
    '''
    This method loads training and testing data as pandas data frames. 


    '''

    # Read file
    df = pd.read_csv(training_data_path, sep=';')
    
    # Sample data
    train=df.sample(frac=frac_)
    test=df.drop(train.index)
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    return train, test

def load_txt_files(source_path):
    '''
    This method  reads all txt files from the source path and stores
    their content into a pandas table (pickle) inside the target path
    '''

    sections = []

    # Iterate over all files in directory
    filenames = [f for f in os.listdir(source_path)] 
        
    for filename in filenames:
            
        file = open(source_path + filename, 'r')
        file_content = file.read()

        page_nr = 1
        section_idx = 0 # Each section is indexed. The filename and the index are a composite key for a section

        for page in file_content.split('\n\n'): # Pages are separated with a double new line
                
            for section in page.split('\n'): # Sections are separated with a single new line
                sections.append([filename, section_idx, page_nr, section])
                section_idx += 1
                
            page_nr += 1

    df = pd.DataFrame(sections, columns=['report_id', 'section_index', 'page_number', 'section_text'])

    return df
 

