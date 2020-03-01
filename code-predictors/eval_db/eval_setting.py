from typing import List

from eval_db.database import Database

INSERT_STATEMENT = "INSERT INTO settings ('id', 'agent', 'track', 'time', 'weather') values (?,?,?,?,?);"


class Setting:
    def __init__(self, id: int, agent: str, track: str, time: str, weather: str):
        self.id: int = id
        self.agent: str = agent
        self.track: str = track
        self.time: str = time
        self.weather: str = weather

    def get_folder_name(self):
        folder_time = self.time
        if self.time == "DayOnly" and self.weather == "Sunny":
            return self.agent + "-" + self.track + "-Normal"
        if self.time == "DayOnly":
            folder_time = ""
        folder_weather = self.weather
        if self.weather == "Sunny":
            folder_weather = ""
        return self.agent + "-" + self.track + "-" + folder_time + folder_weather

    def insert_into_db(self, db: Database) -> None:
        db.cursor.execute(INSERT_STATEMENT,
                          (self.id, self.agent, self.track, self.time, self.weather))


def get_all_settings(db: Database) -> List[Setting]:
    cursor = db.cursor.execute('select * from settings')
    var = cursor.fetchall()
    result = []
    for db_record in var:
        setting = Setting(id=db_record[0], agent=db_record[1], track=db_record[2], time=db_record[3],
                          weather=db_record[4])
        result.append(setting)
    return result
