'''
This module contains a tool kit for transforming text in dataframes in the app_model_validation.py file
'''

import pandas as pd

import langid

def filter_short_text(df, text_col, minimum_length):
    '''
    Filter texts from a pandas dataframe which contains less than minimum_length characters

    # Parameters
        df: A pandas data frame
        text_col: the name of the column containing the texts
        minimum_length: minimum number of characters to keep a text
    '''
    df_res = df[df[text_col].apply(lambda text: len(text) >= minimum_length)]
    return df_res

def add_language(df, source_col, target_col):
    '''
    This function adds a language column to a pandas dataframe

    # Parameters
        df: A pandas data frame
        source_col: The name of the column containing the texts
        target_col: The name of the new column which contains the detected language
    '''
    language_tags = []

    for text in df[source_col]:

        try:
            language = langid.classify(text)
            language_tags.append(language[0])
        except:
            language_tags.append(None)

    df[target_col] = language_tags
    return df

def count_labeled_sections(df, labels):
    '''
    Thiis function summarizes the numer of sections with specific labels
    '''
    result = []

    for report_id in pd.Series.unique(df['report_id']):
        df_tmp = df[df['report_id'] == report_id]
        report_result = [report_id, len(df_tmp)]

        for label in labels:
            report_result.append(len(df_tmp[df_tmp[label] == label]))

        
        result.append(report_result)

    return pd.DataFrame(result, columns=['report_id', 'num_sections'] + labels)



