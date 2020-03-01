import logging
from typing import List

import utils_logging
from eval_db.database import Database

INSERT_STATEMENT = "INSERT INTO single_image_based_distances ('setting_id', 'row_id', 'is_crash', 'vae_loss', 'cae_loss', 'dae_loss', 'sae_loss', 'deeproad_loss') values (?,?,?,?,?,?,?,?);"
logger = logging.Logger("SingleImgDistance")
utils_logging.log_info(logger)


class SingleImgDistance:
    def __init__(self, setting_id: int, row_id: int, is_crash: bool, vae_loss: float, cae_loss: float, dae_loss: float,
                 sae_loss: float, deeproad_loss: float):
        self.setting_id = setting_id
        self.row_id = row_id
        self.is_crash = is_crash
        self.vae_loss = vae_loss
        self.cae_loss = cae_loss
        self.dae_loss = dae_loss
        self.sae_loss = sae_loss
        self.deeproad_loss = deeproad_loss
        self.true_label = None
        self.count_to_crash = None

    def insert_into_db(self, db: Database) -> None:
        int_is_crash = 0
        if self.is_crash:
            int_is_crash = 1
        db.cursor.execute(INSERT_STATEMENT,
                          (self.setting_id, self.row_id, int_is_crash, self.vae_loss, self.cae_loss, self.dae_loss,
                           self.sae_loss, self.deeproad_loss))

    def loss_of(self, ad_name: str):
        if ad_name == "vae":
            return self.vae_loss
        if ad_name == "dae":
            return self.dae_loss
        if ad_name == "cae":
            return self.cae_loss
        if ad_name == "sae":
            return self.sae_loss
        if ad_name == "deeproad":
            return self.deeproad_loss
        logger.error("Unknown ad_name")
        assert False


def load_all_for_setting(db: Database, setting_id: int) -> List[SingleImgDistance]:
    cursor = db.cursor.execute('select * from single_image_based_distances where setting_id=? ' +
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
        distance_object = SingleImgDistance(
            setting_id=db_record[0], row_id=db_record[1], is_crash=is_crash, vae_loss=db_record[3],
            cae_loss=db_record[4], dae_loss=db_record[5], sae_loss=db_record[6], deeproad_loss=db_record[7])
        result.append(distance_object)
    return result


def update_true_label_on_db(db: Database, records: List[SingleImgDistance]):
    for record in records:
        insertable = (record.true_label, record.count_to_crash, record.setting_id, record.row_id)
        db.cursor.execute('update single_image_based_distances set true_label=?, count_to_crash=? ' +
                          'where main.single_image_based_distances.setting_id = ?' +
                          ' and main.single_image_based_distances.row_id = ? ',
                          insertable)
