import logging
from typing import Tuple

import numpy
from sklearn.metrics import auc

import utils_logging
from eval_db import eval_prec_recall, eval_window
from eval_db.database import Database
from eval_db.eval_prec_recall import PrecisionRecallAnalysis
from eval_scripts import db_path

CALC_AUROC = True
CALC_PREC_RECALL = False

AUROC_CALC_SAMPLING_FACTOR = 20  # 1 = No Sampling, n = 1/n of the losses are considered

THRESHOLDS = {
    "vae": {"0.68": 27.07115916991946, "0.9": 38.120301897480466, "0.95": 43.86094312127071, "0.99": 56.038196376888564,
            "0.999": 71.95140165124738, "0.9999": 86.90662166303552, "0.99999": 101.28409503970909},
    "cae": {"0.68": 13.376903121103881, "0.9": 19.26048873572004, "0.95": 22.34506281943677, "0.99": 28.93117824199429,
            "0.999": 37.60016052364154, "0.9999": 45.79148090225994, "0.99999": 53.695420701648914},
    "dae": {"0.68": 40.272194049324966, "0.9": 51.9341337226326, "0.95": 57.77528181318816, "0.99": 69.83010882684272,
            "0.999": 85.09783843952644, "0.9999": 99.10057054295928, "0.99999": 112.3336694381103},
    "deeproad": {"0.68": 3374.5632471550516, "0.9": 5363.3898674739285, "0.95": 6447.5932314257825,
                 "0.99": 8826.728405803366, "0.999": 12050.394098718236, "0.9999": 15160.981798506993,
                 "0.99999": 18204.28532004644},
    "sae": {"0.68": 29.361018856328876, "0.9": 40.45529729327319, "0.95": 46.16822971023903, "0.99": 58.20754461663782,
            "0.999": 73.82588456863898, "0.9999": 88.42246606777259, "0.99999": 102.40142769284085}
}

SEQ_THRESHOLDS = {
    "lstm": {"0.68": 0.03967471005673909, "0.9": 0.04960214235753555, "0.95": 0.054508963524284165,
             "0.99": 0.0645317369955604, "0.999": 0.07707193970593226, "0.9999": 0.08845992912613872,
             "0.99999": 0.0991450074191446}
}

logger = logging.Logger("Calc_Precision_Recall")
utils_logging.log_info(logger)


def calc_precision_recall():
    logger.warning("ATTENTION: Thresholds are hardcoded. Copy-paste after recalculating thresholds " +
                   "(hence, after each training of the models)!")

    db_name = db_path.DB_PATH
    db = Database(name=db_name, delete_existing=False)
    eval_prec_recall.remove_all_from_prec_recall(db=db)

    for ad_name, ad_thresholds in THRESHOLDS.items():
        _eval(ad_name=ad_name, ad_thresholds=ad_thresholds, db=db)
    db.commit()

    for ad_name, ad_thresholds in SEQ_THRESHOLDS.items():
        _eval(ad_name=ad_name, ad_thresholds=ad_thresholds, db=db)
    db.commit()


def _eval(ad_name, ad_thresholds, db):
    # TODO Store auc_prec_recall in db as well
    auroc, auc_prec_recall = calc_auroc_and_auc_prec_recall(db=db, ad_name=ad_name)
    for threshold_type, threshold in ad_thresholds.items():
        precision_recall_analysis = create_precision_recall_analysis(ad_name=ad_name,
                                                                     auroc=auroc, auc_prec_recall=auc_prec_recall,
                                                                     db=db, threshold=threshold,
                                                                     threshold_type=threshold_type)
        eval_prec_recall.insert_into_db(db=db, precision_recall=precision_recall_analysis)
    db.commit()


def create_precision_recall_analysis(ad_name, auroc, auc_prec_recall, db, threshold, threshold_type):
    true_positives = eval_window.get_true_positives_count(db=db, ad_name=ad_name, threshold=threshold)
    false_positives = eval_window.get_false_positives_count_ignore_subsequent(db=db, ad_name=ad_name,
                                                                              threshold=threshold)
    true_negatives = eval_window.get_true_negatives_count(db=db, ad_name=ad_name, threshold=threshold)
    false_negatives = eval_window.get_false_negatives_count(db=db, ad_name=ad_name, threshold=threshold)
    precision_recall_analysis = PrecisionRecallAnalysis(anomaly_detector=ad_name,
                                                        threshold_type=threshold_type,
                                                        threshold=threshold,
                                                        true_positives=true_positives,
                                                        false_positives=false_positives,
                                                        true_negatives=true_negatives,
                                                        false_negatives=false_negatives,
                                                        auroc=auroc,
                                                        auc_prec_recall=auc_prec_recall
                                                        )
    return precision_recall_analysis


# Method also used by auroc plotter
def _calc_auc_roc(false_positive_rates, true_positive_rates):
    pass


def calc_auroc_and_auc_prec_recall(db: Database, ad_name: str) -> Tuple[float, float]:
    labels_ignore_this, losses_list = eval_window.get_all_losses_and_true_labels_for_ad(db=db, ad_name=ad_name)
    false_positive_rates = []
    true_positive_rates = []
    precisions = []
    f1s = []
    logger.info("Calc auc-roc for " + ad_name + " based on " + str(
        len(losses_list) / AUROC_CALC_SAMPLING_FACTOR) + " thresholds. Sampling factor: " + str(
        AUROC_CALC_SAMPLING_FACTOR))
    i = 0
    losses_list.sort()
    losses_list = losses_list[::AUROC_CALC_SAMPLING_FACTOR]
    for loss in losses_list:
        i = i + 1
        if i % 100 == 0:
            logger.info("---> " + str(i) + " out of " + str(len(losses_list)))
        # Temporary, non persisted precision_recall_analysis to calculate TPR and FPR
        precision_recall_analysis = create_precision_recall_analysis(ad_name=ad_name,
                                                                     auroc=None,
                                                                     auc_prec_recall=None,
                                                                     db=db,
                                                                     threshold=loss,
                                                                     threshold_type=None)

        fpr = precision_recall_analysis.false_positive_rate
        tpr = precision_recall_analysis.recall
        prec = precision_recall_analysis.prec
        false_positive_rates.append(fpr)
        true_positive_rates.append(tpr)
        precisions.append(prec)
    false_positive_rates = numpy.asarray(false_positive_rates)
    true_positive_rates = numpy.asarray(true_positive_rates)
    precisions = numpy.asarray(precisions)
    auc_roc = _calc_auc(x=false_positive_rates, y=true_positive_rates)
    auc_prec_recall = _calc_auc(x=true_positive_rates, y=precisions)
    return auc_roc, auc_prec_recall


def get_threshold(ad_name: str, threshold_type: str) -> float:
    if ad_name in THRESHOLDS:
        if threshold_type in THRESHOLDS[ad_name]:
            return THRESHOLDS[ad_name][threshold_type]
    if ad_name in SEQ_THRESHOLDS:
        if threshold_type in SEQ_THRESHOLDS[ad_name]:
            return SEQ_THRESHOLDS[ad_name][threshold_type]
    logger.error("Combination of ad " + ad_name + " and threshold type " + threshold_type + "not known")
    assert False


def _calc_auc(x, y):
    sorted_ids = x.argsort()
    sorted_x = x[sorted_ids]
    co_sorted_y = y[sorted_ids]
    return auc(x=sorted_x, y=co_sorted_y)


if __name__ == '__main__':
    calc_precision_recall()
