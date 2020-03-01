import logging

import utils_logging
from eval_db.database import Database
from eval_scripts import db_path, b_precision_recall_auroc
from eval_scripts.utils import threshold_independent_plotters

logger = logging.Logger("c_timeline")
utils_logging.log_info(logger)

if __name__ == '__main__':
    db_name = db_path.DB_PATH
    db = Database(name=db_name, delete_existing=False)
    # Temporary security check to make sure we don't generate based on the wrong db for the paper...
    assert db_path.DB_PATH == "../../models/trained-anomaly-detectors/20190821-ALL-MODELS-MODIFIED_TRACKS.sqlite"
    if b_precision_recall_auroc.AUROC_CALC_SAMPLING_FACTOR != 1:
        logger.warning("Sampling is >1, this is good for testing, but must not be the case for final version graphs")

    # threshold_dependent_reaction_plotter = ThresholdDependentReactionTimePlotter(db=db)
    # threshold_dependent_reaction_plotter.compute_and_plot()

    reaction_plotter = threshold_independent_plotters.ReactionTimePlotter(db=db)
    reaction_plotter.compute_and_plot()

    # k_plotter = threshold_independent_plotters.KSizePlotter(db=db)
    # k_plotter.compute_and_plot()
