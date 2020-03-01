from eval_scripts import db_path, b_precision_recall_auroc, a_set_true_labels

DBS = [
    "../../models/trained-anomaly-detectors/20190821-ALL-MODELS-MODIFIED_TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190821-ALL-MODELS-NORMAL-TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190819-DAVE2-MODIFIED-TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190819-DAVE2-NORMAL-TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190820-CHAUFFEUR-MODIFIED-TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190821-CHAUFFEUR-NORMAL-TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190822-EPOCH-MODIFIED-TRACKS.sqlite",
    "../../models/trained-anomaly-detectors/20190822-EPOCH-NORMAL-TRACKS.sqlite"
]


def recalc_all():
    for db in DBS:
        db_path.DB_PATH = db
        if db == '../../models/trained-anomaly-detectors/20190821-ALL-MODELS-MODIFIED_TRACKS.sqlite':
            b_precision_recall_auroc.AUROC_CALC_SAMPLING_FACTOR = 20
        else:
            b_precision_recall_auroc.AUROC_CALC_SAMPLING_FACTOR = 1
        a_set_true_labels.set_true_labels()
        b_precision_recall_auroc.calc_precision_recall()

if __name__ == '__main__':
    recalc_all()