#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Start the conversation DEMO service
How to run:
    Quick start:
        `python ./deploy/run.py`
    Deploy start:
        `gunicorn -b 0.0.0.0:8888 deploy.run:app --threads 4`
"""
import os
import sys
import json
from flask import Flask, request, render_template

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from deploy.config import get_config
from deploy.utils import DeployError
from deploy.server import ServerCtrl

# load config file
dep_conf = get_config()
app_name = dep_conf['net']['app_name'].strip('/')
app_name = '/' + app_name if app_name else ''

# server control obj
ctrl_server = ServerCtrl(**dep_conf)

# flask app
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True


def get_params_from_request(gp_reqt):
    in_dict = gp_reqt.args.to_dict()
    if gp_reqt.method == 'POST':
        in_dict_post = {}
        if 'application/json' in gp_reqt.headers.environ['CONTENT_TYPE']:
            in_dict_post = gp_reqt.json
        elif 'application/x-www-form-urlencoded' in gp_reqt.headers.environ['CONTENT_TYPE']:
            in_dict_post = gp_reqt.form.to_dict()
        for (key, value) in in_dict_post.items():
            in_dict[key] = value
    return in_dict


@app.route('%s/<fun>' % app_name, methods=['GET', 'POST'])
def net_function(fun):
    params = get_params_from_request(request)
    ret = {}
    try:
        # clear expire session every time
        del_tokens = ctrl_server.on_clear_expire()

        if fun == 'models':
            ret = ctrl_server.on_models()
        elif fun == 'register':
            ret = ctrl_server.on_register(**params)
        elif fun == 'close':
            ret = ctrl_server.on_close(**params)
        elif fun == 'clear_expire':
            ret = del_tokens
        elif fun == 'response':
            ret = ctrl_server.on_response(**params)
        elif fun == 'modify_last':
            ret = ctrl_server.on_modify_last(**params)
        elif fun == 'rollback':
            ret = ctrl_server.on_rollback(**params)
        else:
            raise DeployError('Unknow funtion \'%s\'' % fun)
    except Exception as e:
        err_msg = 'There are some errors in the operation. %s' % str(e)
        if isinstance(e, DeployError):
            err_msg = str(e)
        elif isinstance(e, TypeError):
            err_msg = 'Input parameters incorrect for function \'%s\'.' % fun
        ret = {'status': 'error', 'error_msg': err_msg}
    else:
        ret.setdefault('status', 'ok')
    finally:
        ret = json.dumps(ret, ensure_ascii=False)
    return ret

@app.route("/dialog")
def dialog():
    return render_template("dialog.html")

@app.route("/dialog_eg")
def dialog_eg():
    return render_template("dialog_eg.html")


if __name__ == '__main__':
    # gunicorn deploy.run:app --threads 4
    app.run(host='127.0.0.1', port=dep_conf['net']['port'], debug=True)
