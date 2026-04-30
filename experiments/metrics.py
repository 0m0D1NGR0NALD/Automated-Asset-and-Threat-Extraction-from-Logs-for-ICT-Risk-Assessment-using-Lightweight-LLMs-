from sklearn.metrics import precision_recall_fscore_support, mean_absolute_error
import numpy as np

class MetricsCalculator:
    @staticmethod
    def classification_metrics(y_true, y_pred, average='weighted'):
        prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average)
        return {'precision': prec, 'recall': rec, 'f1': f1}
    
    @staticmethod
    def regression_mae(y_true, y_pred):
        return mean_absolute_error(y_true, y_pred)