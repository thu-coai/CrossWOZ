from enum import Enum
from peewee import *

from ..models import *
from ..redis import rd, with_lock


class Status(Enum):
    FREE = 0
    IN_QUEUE = 1
    MATCHED = 2


class RedisQueue:
    def __init__(self, name):
        self.name = name

    @with_lock("matching")
    def __len__(self):
        return rd.llen('queue:{}'.format(self.name))

    @with_lock("matching")
    def push(self, v):
        rd.rpush('queue:{}'.format(self.name), v)

    @with_lock("matching")
    def pop(self):
        return int(rd.lpop('queue:{}'.format(self.name)))

    @with_lock("matching")
    def remove(self, v):
        rd.lrem('queue:{}'.format(self.name), 0, v)


def create_room(system, client):
    user0 = User.get(User.id == system)
    user1 = User.get(User.id == client)
    task = Task.select().where(Task.finished == False).order_by(fn.Random()).get()
    print(task.content)
    task.finished = True
    task.save()
    room = Room.create(
        task=task,
        user0=user0,
        user1=user1,
        status=Room.Status.RUNNING
    )


@with_lock("matching")
def get_status(uid):
    s = rd.get('user_status:{}'.format(uid)) or 0
    return Status(int(s))


@with_lock("matching")
def set_status(uid, status: Status):
    s = status.value
    rd.set('user_status:{}'.format(uid), s)
