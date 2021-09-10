import bcrypt

from flask import redirect, abort, g
from functools import wraps

from .models import *


def resetdb():
    all_tables = [
        User, Task, Room, Message
    ]
    db.drop_tables(all_tables)
    db.create_tables(all_tables)

    username = 'root'
    password = 'root'
    password_hash = bcrypt.hashpw(str.encode(password), bcrypt.gensalt())
    User.create(username=username, password_hash=password_hash, is_admin=True)


def login_required(f):
    @wraps(f)
    def decorated_f(*args, **kwargs):
        if not g.me:
            return redirect('/login')
        return f(*args, **kwargs)

    return decorated_f


def admin_required(f):
    @wraps(f)
    def decorated_f(*args, **kwargs):
        if not g.me or not g.me.is_admin:
            return redirect('/')
        return f(*args, **kwargs)

    return decorated_f


def room_guard(f):
    @wraps(f)
    def decorated_f(room_id, **kwargs):
        if not g.me:
            return redirect('/')
        try:
            room = Room.get(Room.id == room_id)
            if 'role' in kwargs:
                role = kwargs['role']
                assert role in [0, 1]
                assert (g.me == (room.user0 if role == 0 else room.user1))
        except:
            return abort(404)
        if room.status != Room.Status.RUNNING:
            return redirect('/')
        return f(room, **kwargs)

    return decorated_f
