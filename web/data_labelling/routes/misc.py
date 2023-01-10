import os
import json
from peewee import prefetch, fn
from datetime import datetime, date
from flask import Blueprint, render_template, request, url_for, jsonify

from ..utils import *

bp = Blueprint('misc_bp', __name__)


@bp.route('/login')
def login():
    return render_template('login.html') if not g.me else redirect('/')


@bp.route('/register')
def register():
    return render_template('register.html') if not g.me else redirect('/')


@bp.route('/')
@login_required
def index():
    return redirect('/match') if not g.me.is_admin else redirect('/admin')


@bp.route('/num-tasks-unfinished')
@login_required
def num_tasks_unfinished():
    num = Task.select().where(Task.finished == False).count()
    return jsonify(num)


@bp.route('/import-all', methods=['POST'])
@admin_required
def import_all():
    basepath = 'data_labelling/results/input'
    count = 0

    def import_one(path):
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        def f(options):
            items = []
            for goal in options['goals']:
                field = goal['领域']
                _id = goal['id']
                kv = goal.get('约束条件', []) + goal.get('需求信息', []) + goal.get('预订信息', [])
                for k, v in kv:
                    items.append([_id, field, k, v])
            options['items'] = items
            return options

        count = 0
        for x in data:
            Task.create(content=f(x))
            count += 1
        return count

    for filename in os.listdir(basepath):
        if not filename.endswith('.json'):
            continue
        fullpath = os.path.join(basepath, filename)
        try:
            count += import_one(fullpath)
        except Exception as e:
            print(repr(e))
        try:
            os.remove(fullpath)
        except:
            pass
    return str(count)


@bp.route('/export-all', methods=['POST'])
@admin_required
def export_all():
    basepath = 'data_labelling/results/output'
    n = Task.select(fn.Max(Task.id)).scalar()
    step = 100

    class JSONEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, datetime):
                return obj.strftime('%Y-%m-%d %H:%M:%S')
            elif isinstance(obj, date):
                return obj.strftime('%Y-%m-%d')
            else:
                return json.JSONEncoder.default(self, obj)
    count = 0
    files = []
    for i in range(0, n, step):
        j = min(i + step, n)
        messages = Message.select().order_by(Message.id)
        rooms = Room.select().where((Room.status_code == Room.Status.SUCCESS.value) & (Room.task > i) & (Room.task <= j)).order_by(Room.task)
        rooms = prefetch(rooms, messages)
        data = []
        for room in rooms:
            data.append({
                'task': room.task_id,
                'user': [room.user0_id, room.user1_id],
                'messages': list(map(lambda msg: dict({
                    'role': msg.role,
                    'content': msg.content,
                    'payload': msg.payload,
                    'created_at': msg.created_at
                }), room.messages)),
                'created_at': room.created_at
            })
            count += 1
        fullpath = os.path.join(basepath, '%s.json' % j)
        with open(fullpath, 'w', encoding='utf-8') as f:
            json.dump(data, f, cls=JSONEncoder, indent=4, ensure_ascii=False)
        files.append(fullpath)
    zip_file = os.path.join(basepath, 'all.zip')
    try:
        os.remove(zip_file)
    except:
        pass
    os.system('zip -j %s %s' % (zip_file, ' '.join(files)))
    return str(count)

@bp.route('/remove-waiting-tasks', methods=['POST'])
@admin_required
def remove_waiting_tasks():
    Task.delete().where(Task.id.not_in(Room.select(Task.id).where(Room.task == Task.id))).execute()
    return 'OK'
