#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""load deployment settings from config file

How to edit config file:
Profiles are written in standard json format.
E.g:
```json
{
  "net":
  {
    "port": 8787,           // (not default), Backend service open port
    "app_name": "convlab2"      // (not default), Service access interface path name
    "session_time_out": 300 // (default as 600), The longest life cycle (seconds) in which a session is idle
  },

  "nlu":                    // (Can not be empty), models list of nlu module
  {
    "svm-cam":              // The uniquely identifies of an nlu model. User named.
    {
      "class_path": "convlab2.nlu.svm.camrest.nlu.SVMNLU",  // (not default), Target model class relative path
      "data_set": "camrest",                            // (not default), The data set used by the model
      "ini_params": {"mode": "usr"},                    // (default as {}), The parameters required for the class to be instantiated
      "model_name": "svm",                              // (default as model key), Model name displayed on the front end
      "max_core": 2,                                    // (default as 1), The maximum number of backgrounds allowed for this model to start.
                                                        //                  Recommended to set to 1, or not set.
      "preload": True                                   // (default as true), If false, this model is not preloaded
      "enable": true                                    // (default as true), If false, the system will ignore this configuration
    },

    "my-model":
    {
      "class_path": "convlab2.nlu.svm.multiwoz.nlu.xxx",
      "data_set": "multiwoz"
    }
  },

  "dst":                    // (Can not be empty), models list of dst module
  {
    ...
  },

  "policy":                 // (Can not be empty), models list of policy module
  {
    ...
  },false

  "nlg":                    // (Can not be empty), models list of nlg module
  {
    ...
  }
}
```
"""
import os
import sys
import json


def load_config_file(filepath: str = None) -> dict:
    """
    load config setting from json file
    :param filepath: str, dest config file path
    :return: dict,
    """
    if not isinstance(filepath, str):
        filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), 'dep_config.json'))

    # load
    with open(filepath, 'r', encoding='UTF-8') as f:
        conf = json.load(f)
    assert isinstance(conf, dict), 'Incorrect format in config file \'%s\'' % filepath

    # check sections
    for sec in ['net', 'nlu', 'dst', 'policy', 'nlg']:
        assert sec in conf.keys(), 'Missing \'%s\' section in config file \'%s\'' % (sec, filepath)

    # check net
    conf['net'].setdefault('app_name', '')
    conf['net'].setdefault('session_time_out', 600)
    assert isinstance(conf['net'].get('port', None), int), 'Incorrect key \'net\'->\'port\' in config file \'%s\'' % filepath
    assert isinstance(conf['net'].get('app_name', None), str), 'Incorrect key \'net\'->\'app_name\' in config file \'%s\'' % filepath

    # check modules
    for module in ['nlu', 'dst', 'policy', 'nlg']:
        conf[module] = {key: info for (key, info) in conf[module].items() if info.get('enable', True)}
        assert conf[module], '\'%s\' module can not be empty in config file \'%s\'' % (module, filepath)

        # check every model
        for model in conf[module].keys():
            assert isinstance(conf[module][model], dict), 'Incorrect type for \'%s\'->\'%s\' in config file \'%s\'' % (
            module, model, filepath)
            # set default
            conf[module][model].setdefault('ini_params', {})
            conf[module][model].setdefault('model_name', model)
            conf[module][model].setdefault('max_core', 1)
            conf[module][model].setdefault('preload', True)
            conf[module][model]['max_core'] = 1 if conf[module][model]['max_core'] < 1 else conf[module][model]['max_core']
            assert isinstance(conf[module][model].get("class_path", None), str), \
                'Incorrect type for \'%s\'->\'%s\'->\'class_path\' in config file \'%s\'' % (module, model, filepath)
            assert isinstance(conf[module][model].get("data_set", None), str), \
                'Incorrect type for \'%s\'->\'%s\'->\'data_set\' in config file \'%s\'' % (module, model, filepath)

    return conf


def map_class(cls_path: str):
    """
    Map to class via package text path
    :param cls_path: str, path with `tatk` project directory as relative path, separator with `,`
                            E.g  `tatk.nlu.svm.camrest.nlu.SVMNLU`
    :return: class
    """
    pkgs = cls_path.split('.')
    cls = __import__('.'.join(pkgs[:-1]))
    for pkg in pkgs[1:]:
        cls = getattr(cls, pkg)
    return cls


def get_config(filepath: str = None) -> dict:
    """
    The configuration file is used to create all the information needed for the deployment,
    and the necessary security monitoring has been performed, including the mapping of the class.
    :param filepath: str, dest config file path
    :return: dict
    """
    # load settings
    conf = load_config_file(filepath)

    # add project root dir
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

    # reflect class
    from convlab2.util.module import Module
    # NLU
    from convlab2.nlu import NLU
    for (model, infos) in conf['nlu'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, NLU), '\'%s\' is not a %s class' % (cls_path, 'nlu')
        assert issubclass(cls, Module), '\'%s\' is not a %s class' % (cls_path, 'Module')
        conf['nlu'][model]['class'] = cls

    # DST
    from convlab2.dst import DST
    for (model, infos) in conf['dst'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, DST), '\'%s\' is not a %s class' % (cls_path, 'dst')
        assert issubclass(cls, Module), '\'%s\' is not a %s class' % (cls_path, 'Module')
        conf['dst'][model]['class'] = cls

    # Policy
    from convlab2.policy import Policy
    for (model, infos) in conf['policy'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, Policy), '\'%s\' is not a %s class' % (cls_path, 'policy')
        assert issubclass(cls, Module), '\'%s\' is not a %s class' % (cls_path, 'Module')
        conf['policy'][model]['class'] = cls

    # NLG
    from convlab2.nlg import NLG
    for (model, infos) in conf['nlg'].items():
        cls_path = infos.get('class_path', '')
        cls = map_class(cls_path)
        assert issubclass(cls, NLG), '\'%s\' is not a %s class' % (cls_path, 'nlg')
        assert issubclass(cls, Module), '\'%s\' is not a %s class' % (cls_path, 'Module')
        conf['nlg'][model]['class'] = cls

    return conf


if __name__ == '__main__':
    # test
    cfg = get_config()
    nlu_mod = cfg['nlu']['svm-cam']['class'](**cfg['nlu']['svm-cam']['ini_params'])
    dst_mod = cfg['dst']['rule-cam']['class'](**cfg['dst']['rule-cam']['ini_params'])
    plc_mod = cfg['policy']['mle-cam']['class'](**cfg['policy']['mle-cam']['ini_params'])
    nlg_mod = cfg['nlg']['tmp-cam-usr-manual']['class'](**cfg['nlg']['tmp-cam-usr-manual']['ini_params'])
