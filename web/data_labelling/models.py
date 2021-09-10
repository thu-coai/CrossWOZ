import os
import datetime

from enum import Enum
from dotenv import load_dotenv
from playhouse.sqlite_ext import *

__all__ = ['db', 'User', 'Task', 'Room', 'Message']

load_dotenv()

db = SqliteExtDatabase(os.getenv('DATABASE_PATH'))

SQLITE_EXT_JSON1_PATH = os.getenv('SQLITE_EXT_JSON1_PATH')
if SQLITE_EXT_JSON1_PATH:
    db.load_extension(SQLITE_EXT_JSON1_PATH)


class BaseModel(Model):
    class Meta:
        database = db


class User(BaseModel):
    username = CharField(unique=True)
    password_hash = CharField()

    is_admin = BooleanField(default=False)

    tasks_done = IntegerField(default=0)

    created_at = DateTimeField(default=datetime.datetime.now)

    def updateTasksCount(self):
        self.tasks_done = Room.select().where((Room.status_code == Room.Status.SUCCESS.value) & ((Room.user0 == self) | (Room.user1 == self))).count()
        self.save()


class Task(BaseModel):
    content = JSONField()

    created_at = DateTimeField(default=datetime.datetime.now)

    finished = BooleanField(default=False)


class Room(BaseModel):
    task = ForeignKeyField(Task, backref='rooms')

    user0 = ForeignKeyField(User)
    user1 = ForeignKeyField(User)

    status_code = IntegerField()

    created_at = DateTimeField(default=datetime.datetime.now)

    class Status(Enum):
        RUNNING = 0
        SUCCESS = 1
        ABORTED = 2

    @property
    def status(self):
        return Room.Status(self.status_code)

    @status.setter
    def status(self, status):
        assert isinstance(status, Room.Status)
        self.status_code = status.value


class Message(BaseModel):
    room = ForeignKeyField(Room, backref='messages')
    role = IntegerField()

    content = TextField()
    payload = JSONField()

    created_at = DateTimeField(default=datetime.datetime.now)
