import time

from .helpers import *

systems = RedisQueue("systems")
clients = RedisQueue("clients")


def add_user(uid, role):
    if get_status(uid) != Status.FREE:
        return False

    set_status(uid, Status.IN_QUEUE)
    if role == 0:
        systems.push(uid)
    else:
        clients.push(uid)

    return True


def free_user(uid):
    if get_status(uid) == Status.IN_QUEUE:
        set_status(uid, Status.FREE)
        try:
            systems.remove(uid)
        except ValueError:
            pass
        try:
            clients.remove(uid)
        except ValueError:
            pass


def leave_room(uid):
    if get_status(uid) == Status.MATCHED:
        set_status(uid, Status.FREE)


last_update_time = 0.0
UPDATE_MINIMAL_INTERVAL = 1.0


def update():
    global last_update_time
    now = time.time()
    if now - last_update_time >= UPDATE_MINIMAL_INTERVAL:
        last_update_time = now

        while len(systems) > 0 and len(clients) > 0:
            system = systems.pop()
            client = clients.pop()

            set_status(system, Status.MATCHED)
            set_status(client, Status.MATCHED)
            print('matched {} and {}'.format(system, client))
            create_room(system, client)


def num_waiting():
    return len(systems), len(clients)
