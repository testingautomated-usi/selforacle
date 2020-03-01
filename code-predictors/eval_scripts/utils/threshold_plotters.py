import abc
import logging

import numpy
from matplotlib.pylab import plt

import utils_logging
from eval_db.database import Database
from eval_scripts import a_set_true_labels, b_precision_recall_auroc

AD_NAMES = {
    "vae": "vae",
    "dae": "dae",
    "sae": "sae",
    "cae": "cae",
    "deeproad": "deeproad",
    "lstm": "lstm"
}

logger = logging.Logger("ThresholdDependentPlotter")
utils_logging.log_info(logger)


class AbstractThresholdDependentPlotter(abc.ABC):

    def __init__(self, x_axis: numpy.array, x_axis_label: str, file_name: str, threshold_type: str, db: Database):
        self.threshold_type = threshold_type
        self.db = db
        self.x_axis = x_axis
        self.x_axis_label = x_axis_label
        self.data_points = len(x_axis)
        self.file_name = file_name

    @abc.abstractmethod
    def _set_variable_values(self, data_point: int) -> None:
        """
        Modify environment (typically the a script) to take the new variable parameter into account
        :param data_point: The number of times this method was already called before this call
        """
        logger.error("Must be implemented in child class")
        assert False

    def compute_and_plot(self):
        # COMPUTE
        results = {}
        results["TP"] = {}
        results["FP"] = {}
        results["TN"] = {}
        results["FN"] = {}
        results["TPR"] = {}
        results["FPR"] = {}
        results["F1"] = {}
        results["Prec"] = {}

        for i in range(self.data_points):
            logger.info("Treating plot value for measure point " + str(i))
            self._set_variable_values(i)
            # Run A-Script
            a_set_true_labels.set_true_labels()
            for ad in AD_NAMES.keys():
                if not ad in results["TP"]:
                    for ad_results in results.values():
                        ad_results[ad] = []

                threshold = b_precision_recall_auroc.get_threshold(ad_name=ad, threshold_type=self.threshold_type)
                # Temporary for calculations, not stored to db
                analysis = b_precision_recall_auroc.create_precision_recall_analysis(ad_name=ad,
                                                                                     auroc=None, auc_prec_recall=None,
                                                                                     db=self.db,
                                                                                     threshold=threshold,
                                                                                     threshold_type=self.threshold_type)
                results["TP"][ad].append(self.replace_minus_one(analysis.true_positives))
                results["FP"][ad].append(self.replace_minus_one(analysis.false_positives))
                results["TN"][ad].append(self.replace_minus_one(analysis.true_negatives))
                results["FN"][ad].append(self.replace_minus_one(analysis.false_negatives))
                results["TPR"][ad].append(self.replace_minus_one(analysis.recall))
                results["FPR"][ad].append(self.replace_minus_one(analysis.false_positive_rate))
                results["F1"][ad].append(self.replace_minus_one(analysis.f1))
                results["Prec"][ad].append(self.replace_minus_one(analysis.prec))

        # PLOT
        for key, metric_results in results.items():
            self.plot_graph(metric_results, key)

    def plot_graph(self, results, y_label: str):
        for ad_key, ad_label in AD_NAMES.items():
            plt.plot(self.x_axis, numpy.asarray(results[ad_key], ), label=ad_label)
        plt.legend(loc='lower left')
        plt.xlabel(self.x_axis_label)
        plt.ylabel(ylabel=y_label)
        to_print = plt.gcf()
        plt.show()
        file_name_with_metric = y_label + "-" + self.file_name
        to_print.savefig(file_name_with_metric, bbox_inches='tight')

    def replace_minus_one(self, value):
        if value == -1:
            return -0.05
        return value


class ThresholdDependentReactionTimePlotter(AbstractThresholdDependentPlotter):
    start = 101
    stop = 1
    step = 10

    def __init__(self, db: Database):
        x_axis = numpy.arange(- self.start, -self.stop, self.step)
        x_axis_label = "../../number of images before misbehavior"
        file_name = "reaction-time.pdf"
        super().__init__(x_axis=x_axis, x_axis_label=x_axis_label, file_name=file_name, threshold_type="0.99", db=db)

    def _set_variable_values(self, data_point: int) -> None:
        a_set_true_labels.REACTION_TIME = self.start - (data_point * self.step)
