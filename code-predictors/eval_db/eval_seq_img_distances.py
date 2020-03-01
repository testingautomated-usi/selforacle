import logging
from typing import List

import utils_logging
from eval_db.database import Database

INSERT_STATEMENT = "INSERT INTO sequence_based_distances ('setting_id', 'row_id', 'is_crash', 'lstm_loss') values (?,?,?,?);"
logger = logging.Logger("SeqBasedDistance")
utils_logging.log_info(logger)


class SeqBasedDistance:
    def __init__(self, setting_id: int, row_id: int, is_crash: bool, lstm_loss: float):
        self.setting_id = setting_id
        self.row_id = row_id
        self.is_crash = is_crash
        self.lstm_loss = lstm_loss
        self.true_label = None
        self.count_to_crash = None

    def insert_into_db(self, db: Database) -> None:
        int_is_crash = 0
        if self.is_crash:
            int_is_crash = 1
        db.cursor.execute(INSERT_STATEMENT,
                          (self.setting_id, self.row_id, int_is_crash, self.lstm_loss))

    def loss_of(self, ad_name: str):
        if ad_name == "lstm":
            return self.lstm_loss
        logger.error("Unknown ad_name")
        assert False


def load_all_for_setting(db: Database, setting_id: int) -> List[SeqBasedDistance]:
    cursor = db.cursor.execute('select * from sequence_based_distances where setting_id=? ' +
                               'order by row_id',
                               (setting_id,))
    var = cursor.fetchall()
    result = []
    for db_record in var:
        int_is_crash = db_record[2]
        is_crash = False
        if int_is_crash == 1:
            is_crash = True
        elif int_is_crash != 0:
            logger.error("Unknown is_crash bool encoding")
            exit(1)
        distance_object = SeqBasedDistance(setting_id=db_record[0], row_id=db_record[1], is_crash=is_crash,
                                           lstm_loss=db_record[3])
        result.append(distance_object)
    return result


def update_true_label_on_db(db: Database, records: List[SeqBasedDistance]):
    for record in records:
        insertable = (record.true_label, record.count_to_crash, record.setting_id, record.row_id)
        db.cursor.execute('update sequence_based_distances set true_label=?, count_to_crash=? ' +
                      'where main.sequence_based_distances.setting_id = ?' +
                      ' and main.sequence_based_distances.row_id = ? ',
                      insertable)


