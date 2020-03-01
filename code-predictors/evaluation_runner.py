import logging

import numpy

import training_runner
import utils_args
import utils_logging
from eval_db.database import Database
from eval_db.eval_seq_img_distances import SeqBasedDistance
from eval_db.eval_setting import Setting
from eval_db.eval_single_img_distances import SingleImgDistance

SINGLE_IMAGE_ADS = ['SAE', "VAE", 'CAE', "DAE", "DEEPROAD"]

SEQUENCE_BASED_ADS = ["IMG-LSTM"]

logger = logging.Logger("main")
utils_logging.log_info(logger)

EVAL_AGENTS = ["COMMAAI"]
EVAL_TRACKS = ["Track1", "Track2", "Track3"]
# EVAL_TIME = ["DayOnly"]
# EVAL_WEATHER = ["Sunny"]
# TODO Change this every time you want to merge generated tables to compatible start ids
SETTING_START_ID = 3000
EVAL_TIME = ["DayNight", "DayOnly"]
EVAL_WEATHER = ["Fog", "Rain", "Snow", "Sunny"]


def main():
    # Eval Config, change this line to evaluate agains another set
    eval_dir = "../datasets/eval_data/preliminary-runs/"

    train_args = utils_args.load_train_args()
    train_args.always_calc_thresh = False

    for train_data_dir in train_args.data_dir:
        train_dataset_name = training_runner.dataset_name_from_dir(train_data_dir)
        db_name = "../models/trained-anomaly-detectors/" + train_dataset_name + "-based-eval.sqlite"

        # Prepare Database
        db = Database(db_name, True)
        # Prepare Settings
        settings = _create_all_settings(db)

        # Prepare ADs
        train_args.delete_trained = False
        single_img_based_ads, sequence_based_ads = _prepare_ads(train_data_dir, train_args)

        # Evaluate for Single Image Based
        for setting in settings:
            data_dir = eval_dir + setting.get_folder_name()
            if len(single_img_based_ads) > 0:
                handle_single_image_based_ads(db=db, data_dir=data_dir, setting=setting,
                                              single_img_based_ads=single_img_based_ads)

            if len(sequence_based_ads) > 0:
                handle_sequence_based_ads(db=db, data_dir=data_dir, setting=setting,
                                          sequence_based_ads=sequence_based_ads)


def handle_sequence_based_ads(db, data_dir, setting, sequence_based_ads):
    ad_distances = {}
    frame_ids = None
    are_crashes = None
    for ad_name, ad in sequence_based_ads.items():
        logger.info("Calculating losses for " + setting.get_folder_name() + " with ad  " + ad_name)
        x, y, frm_ids, crashes = ad.load_img_paths(data_dir=data_dir, restrict_size=False, eval_data_mode=True)
        assert len(x) == len(y) == len(frm_ids) == len(crashes)
        distances = ad.calc_losses(inputs=x, labels=y, data_dir=data_dir)
        ad_distances[ad_name] = distances
        if frame_ids is None:
            frame_ids = frm_ids
            are_crashes = crashes
    logger.info("Done. Now storing sequence based eval for setting " + setting.get_folder_name())
    store_seq_losses(setting=setting, per_ad_distances=ad_distances, row_ids=frame_ids,
                     are_crashes=are_crashes, db=db)


def handle_single_image_based_ads(db, data_dir, setting, single_img_based_ads):
    ad_distances = {}
    frame_ids = None
    are_crashes = None
    for ad_name, ad in single_img_based_ads.items():
        logger.info("Calculating losses for " + setting.get_folder_name() + " with ad  " + ad_name)
        x, frm_ids, crashes = ad.load_img_paths(data_dir=data_dir, restrict_size=False, eval_data_mode=True)
        assert len(x) == len(frm_ids) == len(crashes)
        distances = ad.calc_losses(inputs=x, labels=None, data_dir=data_dir)
        ad_distances[ad_name] = distances
        if frame_ids is None:
            frame_ids = frm_ids
            are_crashes = crashes
    logger.info("Done. Now storing single img based eval for setting " + setting.get_folder_name())
    store_losses(setting=setting, per_ad_distances=ad_distances, row_ids=frame_ids,
                 are_crashes=are_crashes, db=db)


def store_seq_losses(setting, per_ad_distances, row_ids, are_crashes, db: Database):
    for i in range(len(per_ad_distances["IMG-LSTM"])):
        setting_id = setting.id
        row_id = row_ids[i]
        row_id = row_id.item()
        if are_crashes[i] == 0:
            is_crash = False
        else:
            is_crash = True
        lstm_loss = per_ad_distances["IMG-LSTM"][i]
        to_store = SeqBasedDistance(setting_id=setting_id, row_id=row_id, is_crash=is_crash, lstm_loss=lstm_loss)
        to_store.insert_into_db(db)
        if i % 1000:
            db.commit()
    db.commit()


def store_losses(setting, per_ad_distances, row_ids, are_crashes, db: Database):
    for i in range(len(per_ad_distances["VAE"])):
        setting_id = setting.id
        row_id = row_ids[i]
        row_id = row_id.item()
        if are_crashes[i] == 0:
            is_crash = False
        else:
            is_crash = True
        vae_loss = per_ad_distances["VAE"][i]
        sae_loss = per_ad_distances["SAE"][i]
        cae_loss = per_ad_distances["CAE"][i]
        dae_loss = per_ad_distances["DAE"][i]
        deeproad_loss = per_ad_distances["DEEPROAD"][i]
        deeproad_loss = deeproad_loss.item()
        to_store = SingleImgDistance(setting_id=setting_id, row_id=row_id, is_crash=is_crash, vae_loss=vae_loss,
                                     cae_loss=cae_loss, dae_loss=dae_loss, sae_loss=sae_loss,
                                     deeproad_loss=deeproad_loss)
        to_store.insert_into_db(db)
        if i % 1000:
            db.commit()
    db.commit()


def _prepare_ads(data_dir, train_args):
    single_img_ads = {}
    for ad_name in SINGLE_IMAGE_ADS:
        single_img_ads[ad_name] = training_runner.load_or_train_model(args=train_args, data_dir=data_dir,
                                                                      model_name=ad_name)
    sequence_based_ads = {}
    logger.warning("Enable squence based again")
    for ad_name in SEQUENCE_BASED_ADS:
        sequence_based_ads[ad_name] = training_runner.load_or_train_model(args=train_args, data_dir=data_dir,
                                                                          model_name=ad_name)
    return single_img_ads, sequence_based_ads


def _create_all_settings(db: Database):
    settings = []
    id = SETTING_START_ID
    for agent in EVAL_AGENTS:
        for track in EVAL_TRACKS:
            for time in EVAL_TIME:
                for weather in EVAL_WEATHER:
                    if not (time == "DayOnly" and weather == "Sunny"):
                        setting = Setting(id=id, agent=agent, track=track, time=time, weather=weather)
                        setting.insert_into_db(db=db)
                        id = id + 1
                        settings.append(setting)
    db.commit()
    return settings


if __name__ == '__main__':
    main()
