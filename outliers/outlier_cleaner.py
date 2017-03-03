#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    origin_data = []
    for idx in range(len(predictions)):
        data = (predictions[idx] - net_worths[idx], ages[idx], net_worths[idx])
        origin_data.append(data)

    sorted(origin_data, cmp=lambda x, y: cmp(abs(x[0]), abs(y[0])))

    for idx in range(len(origin_data)):
        if idx <= 0.9 * len(origin_data):
            data = origin_data[idx]
            cleaned_data.append(data)

    return cleaned_data

