
'''
This module contains a tool kit for validating the model performance in the app_model_validation.py file
'''

def calculate_statistics(labels_, labels_pred, label_to_test, texts):
    '''
    This method calculates test statistics for two lists of true and predicted labels.

    Parameters:
       labels_: A list with the test data (true labels)
       labels_pred: A list with predicted labels. Must be same length as labels_
       label_to_test: A string specifying the label on which the validation should focus
       texts: The texts for which the labels were generated. Must be same length as labels_

    #Returns:
        dict: Returns a dict with the entries
                - pos: Number of positives in ground truth
                - neg: Number of negatives in ground truth
                - true_pos: Number of true positives
                - true_pos_entries: The texts for which the predictions where true positives
                - true_neg: Number of true negatives
                - true_neg_entries: The texts for which the predictins where true negatives
                - false_pos: Number of false positives
                - false_pos_entries: The texts for which the predictins where false positives
                - false_neg: Number of false negatives
                - false_neg_entries: The texts for which the predictins where false negatives

    Based on these entries the following performance metrics can be calculated:
        - accuracy: (true_pos + true_neg) / (pos + neg)
        - recall: true_pos / pos
        - false negative rate: false_neg / pos
        - false positive rate: fale_pos / neg
    '''

    result = {} # dict with results
    labels = labels_.copy() # copy labels so they are not altered in non local environment

    # Get number of positives in ground truth
    pos = 0
    for i in range(len(labels)):
        if labels[i] != label_to_test:
            labels[i] = None
        else:
            pos += 1

    # Get number of negatives in ground truth
    result['pos'] = pos
    result['neg'] = len(labels) - pos


    # Iterate over all labels and count true_pos, true_neg, flase_pos and false_neg
    true_pos = 0
    true_pos_entries = []
    true_neg = 0
    true_neg_entries = []
    false_pos = 0
    false_pos_entries = []
    false_neg = 0
    false_neg_entries = []

    for i in range(len(labels)):
        if labels[i] == labels_pred[i] and labels[i] == label_to_test:
            true_pos += 1
            true_pos_entries.append(texts[i])
        elif labels[i] == labels_pred[i] and labels[i] == None:
            true_neg += 1
            true_neg_entries.append(texts[i])
        elif labels[i] != labels_pred[i] and labels[i] == label_to_test:
            false_neg += 1
            false_neg_entries.append(texts[i])
        else:
            false_pos += 1
            false_pos_entries.append(texts[i])

    # Fill results into dict
    result['true_pos'] = true_pos
    result['true_pos_entries'] = true_pos_entries

    result['true_neg'] = true_neg
    result['true_neg_entries'] =true_neg_entries

    result['false_pos'] = false_pos
    result['false_pos_entries'] = false_pos_entries

    result['false_neg'] = false_neg
    result['false_neg_entries'] = false_neg_entries

    return result
