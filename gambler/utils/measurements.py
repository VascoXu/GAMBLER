def translate_measurement(measurement, dataset):
    if dataset == 'eog':
        # EOG data
        return measurement[0]

    if dataset == 'uci_har' or dataset == 'epilepsy':
        # Accelerometer data
        return (measurement[0]**2 + measurement[1]**2 + measurement[2]**2)**0.5
