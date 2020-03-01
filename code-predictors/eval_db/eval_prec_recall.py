from eval_db.database import Database

INSERT_STRING = "INSERT INTO 'prec_recall' ('anomaly_detector', 'threshold_type', 'threshold', 'true_positives', " \
                "'false_positives', 'true_negatives', 'false_negatives', 'prec', 'recall', 'f1', 'num_anomalies'," \
                " 'num_normal', 'auroc', 'false_positive_rate', 'pr_auc') VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)"


class PrecisionRecallAnalysis:
    def __init__(self,
                 anomaly_detector: str,
                 threshold_type: str,
                 threshold: float,
                 true_positives: int,
                 false_positives: int,
                 true_negatives: int,
                 false_negatives: int,
                 auroc: float,
                 auc_prec_recall: float
                 ):
        self.anomaly_detector = anomaly_detector
        self.threshold_type = threshold_type
        self.threshold = threshold
        self.true_positives = true_positives
        self.false_positives = false_positives
        self.true_negatives = true_negatives
        self.false_negatives = false_negatives
        self.prec = self._calculate_precision(true_positives=true_positives, false_positives=false_positives )
        self.recall = self._calculate_recall(true_positives=true_positives, false_negatives=false_negatives )
        self.f1 = self._calculate_f1(precision=self.prec, recall=self.recall)
        self.num_anomalies = true_positives + false_negatives
        self.num_normal = false_positives + true_negatives
        self.auroc = auroc
        self.false_positive_rate = self._calc_false_positive_rate(false_positives=false_positives, true_negatives=true_negatives)
        self.pr_auc = auc_prec_recall

    @staticmethod
    def _calc_false_positive_rate(false_positives, true_negatives):
        if false_positives + true_negatives == 0:
            return -1
        return false_positives / (false_positives + true_negatives)

    @staticmethod
    def _calculate_precision(true_positives: int, false_positives: int) -> float:
        if true_positives + false_positives == 0:
            return -1
        return true_positives / (true_positives + false_positives)

    @staticmethod
    def _calculate_recall(true_positives: int, false_negatives: int) -> float:
        if true_positives + false_negatives == 0:
            return -1
        return true_positives / (true_positives + false_negatives)

    @staticmethod
    def _calculate_f1(precision: float, recall: float):
        if precision == -1 or recall == -1 or precision == 0 or recall == 0:
            return -1
        res = 2 * (precision * recall) / (precision + recall)
        # return - math.log(res, 10)
        return res


def insert_into_db(db: Database, precision_recall: PrecisionRecallAnalysis):
    insertable = (
        precision_recall.anomaly_detector,
        precision_recall.threshold_type,
        precision_recall.threshold,
        precision_recall.true_positives,
        precision_recall.false_positives,
        precision_recall.true_negatives,
        precision_recall.false_negatives,
        precision_recall.prec,
        precision_recall.recall,
        precision_recall.f1,
        precision_recall.num_anomalies,
        precision_recall.num_normal,
        precision_recall.auroc,
        precision_recall.false_positive_rate,
        precision_recall.pr_auc
    )
    db.cursor.execute(INSERT_STRING, insertable)


def get_aurocs(db: Database):
    cursor = db.cursor.execute("SELECT anomaly_detector, auroc FROM prec_recall GROUP BY anomaly_detector, auroc")
    records = cursor.fetchall()
    result = {}
    for record in records:
        result[record[0]] = record[1]
    return result


def remove_all_from_prec_recall(db: Database):
    db.cursor.execute("delete from prec_recall")
