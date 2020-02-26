###############################################################################
# PyDial: Multi-domain Statistical Spoken Dialogue System Software
###############################################################################
#
# Copyright 2015 - 2019
# Cambridge University Engineering Department Dialogue Systems Group
#
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
###############################################################################

import json
import os


class dataset_walker(object):
    def __init__(self, dataListFile, labels=False, dataroot=None):
        self.labels = labels
        self.dataroot = dataroot

        # load dataset (list of calls)
        self.session_list = []
        # for dataset_session_list in self.dataset_session_lists :
        f = open(dataListFile)
        for line in f:
            line = line.strip()
            if (line in self.session_list):
                raise RuntimeError('Call appears twice: %s' % (line))
            self.session_list.append(line)
        f.close()
        print(dataListFile,len(self.session_list))

    def __iter__(self):
        for session_id in self.session_list:
            session_id_list = session_id.split('/')
            session_dirname = os.path.join(self.dataroot,*session_id_list)
            applog_filename = os.path.join(session_dirname,'log.json')
            if (self.labels):
                labels_filename = os.path.join(session_dirname,'label.json')
                if (not os.path.exists(labels_filename)):
                    raise RuntimeError('Cant score : cant open labels file %s' % (labels_filename))
            else:
                labels_filename = None
            print(applog_filename,labels_filename)
            call = Call(applog_filename,labels_filename)
            call.dirname = session_dirname
            yield call
    def __len__(self, ):
        return len(self.session_list)
    

class Call(object):
    def __init__(self,applog_filename,labels_filename):
        self.applog_filename = applog_filename
        self.labels_filename = labels_filename
        f = open(applog_filename)
        self.log = json.load(f)
        f.close()
        if (labels_filename != None):
            f = open(labels_filename)
            self.labels = json.load(f)
            f.close()
        else:
            self.labels = None

    def __iter__(self):
        if (self.labels_filename != None):
            for (log,labels) in zip(self.log['turns'],self.labels['turns']):
                yield (log,labels)
        else: 
            for log in self.log['turns']:
                yield (log,None)
                
    def __len__(self, ):
        return len(self.log['turns'])
