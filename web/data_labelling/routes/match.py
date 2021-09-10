from flask import Blueprint, render_template, jsonify
from flask_socketio import emit

from ..utils import *
from ..models import *
from .. import match_making

bp = Blueprint('match_bp', __name__)


@bp.route('/')
@login_required
def index():
    return render_template('match.html')


@bp.route('/enter-queue/<int:role>', methods=['POST'])
@login_required
def enter_queue(role):
    match_making.update()
    if match_making.add_user(g.me.id, role):
        return 'OK'
    else:
        return '用户已经在队列中', 400


@bp.route('/quit-queue', methods=['POST'])
@login_required
def quit_queue():
    match_making.update()

    uid = g.me.id
    if match_making.get_status(uid) == match_making.Status.IN_QUEUE:
        match_making.free_user(uid)
        return 'OK'
    else:
        return '用户不在队列中', 400


@bp.route('/num-waiting')
@login_required
def num_waiting():
    match_making.update()
    return jsonify(match_making.num_waiting())


@bp.route('/get-room')
@login_required
def get_room():
    match_making.update()

    uid = g.me.id
    if match_making.get_status(uid) == match_making.Status.MATCHED:
        try:
            room = Room.get((Room.status_code == Room.Status.RUNNING.value) &
                            ((Room.user0 == g.me) | (Room.user1 == g.me)))
            role = 0
            if room.user1 == g.me:
                role = 1
            return jsonify({
                'room_id': room.id,
                'role': role
            })
        except:
            return '房间不存在', 400

    return '204', 204
