import abc
import logging

import numpy
from matplotlib.pylab import plt

import utils_logging
from eval_db import eval_window
from eval_db.database import Database
from eval_scripts import a_set_true_labels, b_precision_recall_auroc, db_path

AD_NAMES = {
    "vae": "vae",
    "dae": "dae",
    "sae": "sae",
    "cae": "cae",
    "deeproad": "deeproad",
    "lstm": "lstm"
}

logger = logging.Logger("Plotters")
utils_logging.log_info(logger)


class AbstractThresholdIndependentPlotter(abc.ABC):

    def __init__(self, x_axis: numpy.array, x_axis_label: str, file_name: str, db: Database):
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
        auroc_results = {}
        auc_prec_recall_results = {}
        auc_f1_results = {}
        for i in range(self.data_points):
            logger.info("Treating plot value for measure point " + str(i))
            self._set_variable_values(i)
            # Run A-Script
            a_set_true_labels.set_true_labels()
            for ad in AD_NAMES.keys():
                if not ad in auroc_results:
                    auroc_results[ad] = []
                    auc_prec_recall_results[ad] = []
                    auc_f1_results[ad] = []
                auroc, auc_pr = b_precision_recall_auroc.calc_auroc_and_auc_prec_recall(db=self.db, ad_name=ad)
                auroc_results[ad].append(auroc)
                auc_prec_recall_results[ad].append(auc_pr)

        # PLOT
        self.plot_graph(auroc_results, "AUC-ROC")
        self.plot_graph(auc_prec_recall_results, "AUC-PRC")

    def plot_graph(self, auroc_results, y_label: str):
        for ad_key, ad_label in AD_NAMES.items():
            plt.plot(self.x_axis, numpy.asarray(auroc_results[ad_key], ), label=ad_label)
        plt.legend(loc='lower left')
        plt.xlabel(self.x_axis_label)
        plt.ylabel(ylabel=y_label)
        to_print = plt.gcf()
        plt.show()
        file_name_with_metric = y_label + "-" + self.file_name
        to_print.savefig(file_name_with_metric, bbox_inches='tight')


class ReactionTimePlotter(AbstractThresholdIndependentPlotter):
    start = 101
    stop = 1
    step = 10

    def __init__(self, db: Database):
        x_axis = numpy.arange(- self.start, -self.stop, self.step)
        x_axis_label = "../../number of images before misbehavior"
        file_name = "reaction-time.pdf"
        super().__init__(x_axis, x_axis_label, file_name, db)

    def _set_variable_values(self, data_point: int) -> None:
        a_set_true_labels.REACTION_TIME = self.start - (data_point * self.step)

class KSizePlotter(AbstractThresholdIndependentPlotter):
    start = 1
    stop = 10
    step = 1

    def __init__(self, db: Database):
        x_axis = numpy.arange(self.start, self.stop, self.step)
        x_axis_label = "number of images considered in calculation of time aware loss-score"
        file_name = "../../plots/k-analysis.pdf"
        super().__init__(x_axis, x_axis_label, file_name, db)

    def _set_variable_values(self, data_point: int) -> None:
        eval_window.CALC_SCOPE = data_point + 1

