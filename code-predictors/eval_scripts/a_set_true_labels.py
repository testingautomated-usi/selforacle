import logging
from typing import List

import utils_logging
from eval_db import eval_setting, eval_single_img_distances, eval_seq_img_distances, eval_window
from eval_db.database import Database
from eval_scripts import db_path

NORMAL_LABEL = "normal"

LABEL_GAP = "gap"

ANOMALY_LABEL = "anomaly"

REACTION_LABEL = "reaction"

IGNORE_END_OF_STREAM_LABEL = "ignore_end_of_stream"

HEALING_LABEL = "healing"

MISBEHAVIOR_LABEL = "misbehavior"

REACTION_TIME = 50
ANOMALY_WINDOW_LENGTH = 30
NORMAL_WINDOW_LENGTH = ANOMALY_WINDOW_LENGTH
HEALING_TIME = 60
MAX_CUT_END_LENGTH = REACTION_TIME + ANOMALY_WINDOW_LENGTH
# length of normal-to-crash is REACTION_TIME + ANOMALY_WINDOW_LENGTH (current logic relies on this)

logger = logging.Logger("set_truelabels")
utils_logging.log_info(logger)


def set_true_labels():
    db = Database(name=db_path.DB_PATH, delete_existing=False)

    eval_window.remove_all_stored_records(db=db)
    settings = eval_setting.get_all_settings(db)

    for setting in settings:
        logger.info("labelling windows for setting " + str(setting.id))
        current_window_count = 0

        current_ad_type = "single-img"
        single_img_entries = eval_single_img_distances.load_all_for_setting(db, setting_id=setting.id)
        current_window_count = _set_true_labels(entries=single_img_entries, current_setting_id=setting.id,
                                                current_ad_type=current_ad_type,
                                                current_window_count=current_window_count, db=db)
        eval_single_img_distances.update_true_label_on_db(db=db, records=single_img_entries)

        current_ad_type = "seq"
        single_img_entries = eval_seq_img_distances.load_all_for_setting(db, setting_id=setting.id)
        _set_true_labels(entries=single_img_entries, current_setting_id=setting.id, current_ad_type=current_ad_type,
                         current_window_count=current_window_count, db=db)
        eval_seq_img_distances.update_true_label_on_db(db=db, records=single_img_entries)

    db.commit()


def _remove_true_labels(entries):
    for entry in entries:
        entry.true_label = None


def _detect_misbehaviors(entries):
    for entry in entries:
        if entry.is_crash:
            entry.true_label = MISBEHAVIOR_LABEL


def _detect_healing(entries):
    idx_after_misbehavior_series = _first_after_misbehavior_series(entries=entries)
    for start_index in idx_after_misbehavior_series:
        end_index_excl = start_index + HEALING_TIME
        end_index_excl = min(end_index_excl, len(entries))
        for i in range(start_index, end_index_excl):
            entry = entries[i]
            if entry.true_label is None:
                entry.true_label = HEALING_LABEL
            elif entry.true_label == MISBEHAVIOR_LABEL:
                # Other misbehavior in recovery period. Re-start with next recovery period
                # Resetting end_index_excl to make sure
                #       the break condition which checks for the end of the entries list does not apply
                end_index_excl = -1
                break
            else:
                logger.error("At this point of the label setting, there should only be misbehaviors and None labels")
                assert False
        if end_index_excl == len(entries):
            break


def _reverse_and_detect_and_ignore_last_images(entries) -> bool:
    entries.reverse()
    is_last_entry = True
    count = 0
    for entry in entries:
        if entry.true_label is None:
            entry.true_label = IGNORE_END_OF_STREAM_LABEL
            count = count + 1
            if count == MAX_CUT_END_LENGTH:
                return True
        elif entry.true_label == HEALING_LABEL:
            # Reached healing period of last misbehavior of the stream (looked at in natural order)
            break
        elif entry.true_label == MISBEHAVIOR_LABEL and is_last_entry:
            # Last entry happens to be a misbehavior
            break
        else:
            logger.error(
                "At this point of the label setting, there should only be None, healing or misbehavior (if last frame) labels")
            assert False
        is_last_entry = False
    return False


def _mark_window_with_label(entries, start_inclusive: int, end_excl: int, label: str):
    for i in range(start_inclusive, end_excl):
        entry = entries[i]
        assert entry.true_label is None  # Should not re-label entry
        entry.true_label = label


def _get_window_idx_or_mark_gap(start_inclusive, window_size, entries) -> (bool, int, int):
    if start_inclusive >= len(entries):
        # Window has length 0, hence we give the done command
        return True, -1, -1
    else:
        too_small = len(entries) < start_inclusive + window_size + 1
        end_excl = min(start_inclusive + window_size, len(entries))
        for i in range(start_inclusive, end_excl):
            if entries[i].true_label is not None:
                too_small = True
                end_excl = i
                break
        if too_small:
            _mark_window_with_label(entries=entries, start_inclusive=start_inclusive, end_excl=end_excl,
                                    label=LABEL_GAP)
            return True, start_inclusive, end_excl
        return False, start_inclusive, end_excl


def _detect_and_store_main_blocks(entries, current_setting_id: int, current_ad_type: str, window_count: int,
                                  directly_normal: bool, db: Database):
    # Assert we're still in a reversed image stream
    if len(entries) > 0:
        assert entries[0].row_id > entries[1].row_id
    # Start setting the windows
    first_after_misbehaviour_series = _first_after_misbehavior_series(entries=entries)
    if directly_normal:
        if len(first_after_misbehaviour_series) > 0 and first_after_misbehaviour_series[0] == MAX_CUT_END_LENGTH + 1:
            directly_normal = False
        else:
            first_after_misbehaviour_series.insert(0, MAX_CUT_END_LENGTH + 1)
    for right_after_misbehaviour_series in first_after_misbehaviour_series:
        if not directly_normal:
            # Set reaction window
            abort, reaction_end_excl = attempt_to_set_reaction_window(entries=entries,
                                                                      start_inc=right_after_misbehaviour_series)
            if abort:
                continue
            # Set anomaly window
            abort, anomaly_end_exc, window_count = attempt_to_set_anomaly_window(entries=entries,
                                                                                 start_inc=reaction_end_excl,
                                                                                 window_count=window_count,
                                                                                 setting_id=current_setting_id,
                                                                                 ad_type=current_ad_type,
                                                                                 db=db)
            if abort:
                continue

        # Set normal windows
        if directly_normal:
            directly_normal = False
            anomaly_end_exc = MAX_CUT_END_LENGTH
        window_count = attempt_to_set_normal_windows(entries=entries, anomaly_end_exc=anomaly_end_exc,
                                                     window_count=window_count,
                                                     setting_id=current_setting_id,
                                                     ad_type=current_ad_type,
                                                     db=db)
    return window_count


def _first_after_misbehavior_series(entries) -> List[int]:
    currently_in_misbehavior_serie = False
    result = []
    for i in range(len(entries)):
        entry = entries[i]
        if entry.true_label is None and currently_in_misbehavior_serie:
            result.append(i)
            currently_in_misbehavior_serie = False
        elif entry.true_label == MISBEHAVIOR_LABEL and not currently_in_misbehavior_serie:
            currently_in_misbehavior_serie = True
    return result


def attempt_to_set_normal_windows(entries, anomaly_end_exc: int, window_count: int, setting_id: int,
                                  ad_type: str, db: Database) -> int:
    stop_point = _next_not_none_label_index(entries, anomaly_end_exc, len(entries))
    if stop_point == -1:
        stop_point = len(entries)
    start_point = anomaly_end_exc
    end_point_excl = start_point + NORMAL_WINDOW_LENGTH
    while end_point_excl <= stop_point:
        _mark_window_with_label(entries, start_inclusive=start_point, end_excl=end_point_excl, label=NORMAL_LABEL)
        window_count = _store(entries=entries, start_inc=start_point, end_excl=end_point_excl, window_type="normal",
                              window_count=window_count, setting_id=setting_id, ad_type=ad_type, db=db)
        start_point = end_point_excl
        end_point_excl = start_point + NORMAL_WINDOW_LENGTH
    if end_point_excl > stop_point:
        # Mark remaining space as GAP
        _mark_window_with_label(entries, start_inclusive=start_point, end_excl=stop_point, label=LABEL_GAP)
    return window_count


def attempt_to_set_anomaly_window(entries, start_inc: int, window_count: int, setting_id: int, ad_type: str,
                                  db: Database):
    no_space, anomaly_start, anomaly_end_excl = _get_window_idx_or_mark_gap(start_inclusive=start_inc,
                                                                            window_size=ANOMALY_WINDOW_LENGTH,
                                                                            entries=entries)
    other_label_found = check_and_treat_not_none_labels(anomaly_end_excl=anomaly_end_excl, anomaly_start=anomaly_start,
                                                        entries=entries)
    abort = no_space or other_label_found
    if not abort:
        _mark_window_with_label(entries, start_inclusive=anomaly_start, end_excl=anomaly_end_excl, label=ANOMALY_LABEL)
        window_count = _store(entries=entries, start_inc=anomaly_start, end_excl=anomaly_end_excl,
                              window_type="anomaly", window_count=window_count, setting_id=setting_id, ad_type=ad_type,
                              db=db)
    return abort, anomaly_end_excl, window_count


def _store(entries, start_inc, end_excl, window_type: str, window_count: int, setting_id: int, ad_type: str,
           db: Database) -> int:
    eval_window.store_single_image_windows(setting=setting_id, window_id=window_count,
                                           window_type=window_type, entries=entries, start_inc=start_inc,
                                           end_excl=end_excl, db=db, ad_type=ad_type)
    return window_count + 1


def attempt_to_set_reaction_window(entries, start_inc):
    no_space, reaction_start, reaction_end_excl = _get_window_idx_or_mark_gap(
        start_inclusive=start_inc,
        window_size=REACTION_TIME, entries=entries)
    other_label_found = check_and_treat_not_none_labels(anomaly_end_excl=reaction_end_excl,
                                                        anomaly_start=reaction_start, entries=entries)
    abort = no_space or other_label_found
    if not abort:
        _mark_window_with_label(entries, start_inclusive=reaction_start, end_excl=reaction_end_excl,
                                label=REACTION_LABEL)
    return abort, reaction_end_excl


def check_and_treat_not_none_labels(anomaly_end_excl, anomaly_start, entries):
    other_label_in_window_index = _next_not_none_label_index(entries=entries, start_index_inc=anomaly_start,
                                                             end_index_excl=anomaly_end_excl)

    other_label_found = other_label_in_window_index >= 0
    if other_label_found:
        _mark_window_with_label(entries=entries, start_inclusive=anomaly_start, end_excl=other_label_in_window_index,
                                label=LABEL_GAP)
    return other_label_found


def _next_not_none_label_index(entries, start_index_inc: int, end_index_excl: int):
    assert end_index_excl <= len(entries)
    for i in range(start_index_inc, end_index_excl):
        if entries[i].true_label is not None:
            return i
    return -1


def _integrity_checks_after_calc(entries):
    # Re-create normal order
    entries.reverse()
    _check_order_of_labels(entries)
    _check_frame_length(entries, label=ANOMALY_LABEL, window_length=ANOMALY_WINDOW_LENGTH)
    _check_frame_length(entries, label=NORMAL_LABEL, window_length=NORMAL_WINDOW_LENGTH, allow_subsequent_same=True)
    _check_frame_length(entries, label=REACTION_LABEL, window_length=REACTION_TIME)


def _check_frame_length(entries, label, window_length, allow_subsequent_same=False):
    label_count = 0
    for entry in entries:
        if entry.true_label == label:
            label_count = label_count + 1
            if not allow_subsequent_same:
                assert label_count <= window_length
        else:
            if allow_subsequent_same:
                assert label_count % window_length == 0
            else:
                assert label_count == 0 or label_count == window_length
            label_count = 0


def _check_order_of_labels(entries):
    last_label = ""
    i = 0
    for entry in entries:
        assert entry.true_label is not None
        if last_label == "":
            assert i == 0
        elif entry.true_label == MISBEHAVIOR_LABEL:
            assert last_label == MISBEHAVIOR_LABEL \
                   or last_label == REACTION_LABEL \
                   or last_label == LABEL_GAP \
                   or last_label == HEALING_LABEL
        elif entry.true_label == REACTION_LABEL:
            assert last_label == ANOMALY_LABEL \
                   or last_label == REACTION_LABEL \
                   or last_label == LABEL_GAP \
                   or last_label == HEALING_LABEL
        elif entry.true_label == ANOMALY_LABEL:
            assert last_label == NORMAL_LABEL \
                   or last_label == LABEL_GAP \
                   or last_label == HEALING_LABEL \
                   or last_label == ANOMALY_LABEL
        elif entry.true_label == HEALING_LABEL:
            assert last_label == MISBEHAVIOR_LABEL \
                   or last_label == HEALING_LABEL
        elif entry.true_label == IGNORE_END_OF_STREAM_LABEL:
            assert last_label == HEALING_LABEL \
                   or last_label == NORMAL_LABEL \
                   or last_label == IGNORE_END_OF_STREAM_LABEL
        elif entry.true_label == LABEL_GAP:
            assert last_label == LABEL_GAP \
                   or last_label == HEALING_LABEL

        last_label = entry.true_label
        i = i + 1


def _set_true_labels(entries, current_setting_id: int, current_ad_type: str, current_window_count: int, db: Database):
    _remove_true_labels(entries)
    _detect_misbehaviors(entries)
    _detect_healing(entries)
    directly_normal = _reverse_and_detect_and_ignore_last_images(entries)
    current_window_count = _detect_and_store_main_blocks(entries, current_setting_id=current_setting_id,
                                                         current_ad_type=current_ad_type,
                                                         window_count=current_window_count,
                                                         directly_normal=directly_normal,
                                                         db=db)
    _integrity_checks_after_calc(entries)
    return current_window_count


def _detect_next_misbehavior(entries, start_index_inc) -> int:
    if start_index_inc > len(entries) - 1:
        return -1
    for i in range(start_index_inc, len(entries)):
        if entries[i].is_crash:
            return i
    return -1


if __name__ == '__main__':
    set_true_labels()
