import matplotlib.pyplot as plt
import os, sys
import json
import numpy as np
from pprint import pprint
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from htmlwriter import HTMLWriter




class DomainRecorder():
    def __init__(self, domain):
        self.domain = domain
        self.tot_num = 0
        self.suc_num = 0
        self.pre = 0
        self.rec = 0
        self.f1 = 0
        self.turn_num = 0
        self.turn_num_suc = 0
        self.failed_nlu_da_sys = {}
        self.failed_nlu_da_usr = {}
        self.cycle_start_da = {}
        self.bad_inform = {}
        self.inform_not_reqt = {}
        self.reqt_not_inform = {}

    def add_to_dict(self, da, dict_record):
        if da not in dict_record:
            dict_record[da] = 0
        dict_record[da] += 1
    
    def record(self, suc, inform, fail1, fail2, cycle, turn):
        self.tot_num += 1
        self.suc_num += suc

        TP, FP, FN, _, __, ___ = inform

        try:
            rec = TP / (TP + FN)
        except ZeroDivisionError:
            rec = 0
            pre = 0
            f1 = 0
        try:
            pre = TP / (TP + FP)
            f1 = 2 * pre * rec / (pre + rec)
        except ZeroDivisionError:
            pre = 0
            f1 = 0
        
        self.pre += pre
        self.rec += rec
        self.f1 += f1
        self.turn_num += turn
        if suc == 1:
            self.turn_num_suc += turn

        for da in _:
            self.add_to_dict(da, self.bad_inform)

        for da in __:
            self.add_to_dict(da, self.reqt_not_inform)

        for da in ___:
            self.add_to_dict(da, self.inform_not_reqt)


        for da in fail1:
            if da[0] not in self.failed_nlu_da_sys:
                self.failed_nlu_da_sys[da[0]] = [0, {}]
            self.failed_nlu_da_sys[da[0]][0] += 1
            if da[1] not in self.failed_nlu_da_sys[da[0]][1]:
                self.failed_nlu_da_sys[da[0]][1][da[1]] = 0
            self.failed_nlu_da_sys[da[0]][1][da[1]] += 1

        for da in fail2:
            if da[0] not in self.failed_nlu_da_usr:
                self.failed_nlu_da_usr[da[0]] = [0, {}]
            self.failed_nlu_da_usr[da[0]][0] += 1
            if da[1] not in self.failed_nlu_da_usr[da[0]][1]:
                self.failed_nlu_da_usr[da[0]][1][da[1]] = 0
            self.failed_nlu_da_usr[da[0]][1][da[1]] += 1

        if suc == 0 and len(cycle) > 0:
            da = cycle[-1]
            self.add_to_dict(da, self.cycle_start_da)

    def get_info(self):
        if self.tot_num == 0:
            return 0, 0, 0, 0, 0, 0
        y = [self.cycle_start_da[i] for i in self.cycle_start_da]
        cycle_tot = sum(y)
        tmp = 0 if self.suc_num == 0 else self.turn_num_suc / self.suc_num
        return self.tot_num, self.suc_num / self.tot_num, self.pre / self.tot_num, self.rec / self.tot_num, self.f1 / self.tot_num, cycle_tot, tmp, self.turn_num / self.tot_num
    
    def format_result(self):
        dalist_sys = []
        for da in self.failed_nlu_da_sys:
            tmp = ['-'.join(da), self.failed_nlu_da_sys[da][0], [('-'.join(fda), self.failed_nlu_da_sys[da][1][fda])   for fda in self.failed_nlu_da_sys[da][1] ]]
            dalist_sys.append(tmp)
        
        dalist_usr = []
        for da in self.failed_nlu_da_usr:
            tmp = ['-'.join(da), self.failed_nlu_da_usr[da][0], [('-'.join(fda), self.failed_nlu_da_usr[da][1][fda])   for fda in self.failed_nlu_da_usr[da][1] ]]
            dalist_usr.append(tmp)
        
        cyclist = [('-'.join(da), self.cycle_start_da[da]) for da in self.cycle_start_da]
        badlist = [('-'.join(da), self.bad_inform[da]) for da in self.bad_inform]
        rnilist = [('-'.join(da), self.reqt_not_inform[da]) for da in self.reqt_not_inform]
        inrlist = [('-'.join(da), self.inform_not_reqt[da]) for da in self.inform_not_reqt]

        dalist_sys = sorted(dalist_sys, key=lambda da: da[1], reverse=True)
        dalist_sys = dalist_sys[:10]
        for t in dalist_sys:
            t[2] = sorted(t[2], key=lambda da: da[1], reverse=True)
            t[2] = t[2][:5]

        dalist_usr = sorted(dalist_usr, key=lambda da: da[1], reverse=True)
        dalist_usr = dalist_usr[:10]
        for t in dalist_usr:
            t[2] = sorted(t[2], key=lambda da: da[1], reverse=True)
            t[2] = t[2][:5]

        cyclist = sorted(cyclist, key=lambda da: da[1], reverse=True)
        cyclist = cyclist[:10]
        badlist = sorted(badlist, key=lambda da: da[1], reverse=True)
        badlist = badlist[:10]
        rnilist = sorted(rnilist, key=lambda da: da[1], reverse=True)
        rnilist = rnilist[:10]
        inrlist = sorted(inrlist, key=lambda da: da[1], reverse=True)
        inrlist = inrlist[:10]


        return dalist_sys, dalist_usr, cyclist, badlist, rnilist, inrlist

    


class Reporter():
    def __init__(self, model_name = 'test_model', dataset = 'multiwoz'):
        self.recorders = {}
        self.modelname = model_name
        self.dataset = dataset

    def split_domain(self, domain, das):
        ret = []
        for i in das:
            if len(i) > 0 and i[0][1].lower() == domain:
                for da in i:
                    tmp = (da[0], da[1], da[2], da[3])
                    if tmp not in ret:
                        ret.append(tmp)

        return ret

    def split_domain_nlu(self, domain, das):
        ret = []
        for i in das:
            if len(i[0]) > 0 and i[0][1].lower() == domain:
                if i not in ret:
                    da1 = (i[0][0], i[0][1], i[0][2], i[0][3])
                    da2 = (i[1][0], i[1][1], i[1][2], i[1][3])
                    ret.append((da1, da2))
        return ret

    def record(self, domain, suc, inform, fail1, fail2, cycle, turn):
        if domain not in self.recorders:
            self.recorders[domain] = DomainRecorder(domain)
        fail1 = self.split_domain_nlu(domain, fail1)
        fail2 = self.split_domain_nlu(domain, fail2)
        cycle = self.split_domain(domain, cycle)
        turn_num = turn.count(domain) * 2
        self.recorders[domain].record(suc, inform, fail1, fail2, cycle, turn_num)

    def report(self, com, suc, pre, rec, f1, turn_suc, turn):
        if not os.path.exists('results/'):
            os.mkdir('results')
        
        os.chdir('results/')
        
        if not os.path.exists(self.modelname+'/'):
            os.mkdir(self.modelname)
        
        os.chdir(os.path.pardir)
        
        
        

        writer = HTMLWriter('results/%s/report_%s.html'%(self.modelname, self.dataset))

        writer.write_title('Test Report')
        writer.write_line('Model Name: %s'%self.modelname)
        writer.write_line('Dataset: %s'%self.dataset)
        writer.write_line('Time: %s'%(time.strftime("%Y-%m-%d %H:%M:%S")))


        
        writer.write_line('Overall Results')
        writer.write_metric(suc, pre, rec, f1, turn_suc, turn)

        writer.write_png()

        _, __ = self.format_result()
        writer.report_HTML(_, __)

        writer.write_dialog_loop_png(self.modelname)

        for domain in self.recorders:
            _, suc, pre, rec, f1, _, turn_suc, turn = self.recorders[domain].get_info()
            _, __, ___, ____, _____, ______ = self.recorders[domain].format_result()
            writer.write_domain(domain,suc, pre, rec, f1, turn_suc, turn,  _, __, ___, ____, _____, ______)

        writer.write_done()


        self.plot()
        self.plot_freq()

    def plot_pi(self, y, labels):

        s = sum(y)
        sizes = [i * 100 / s for i in y]

        patches,l_text,p_text = plt.pie(
            sizes, 
            autopct='%1.1f%%', #显示百分比方式
            shadow=False, #阴影效果
            startangle=90,
            labels = labels,
            pctdistance=0.7
        )
        plt.axis('equal')
        for t in l_text:
            t.set_size(25)
        for t in p_text:
            t.set_size(20)

    def plot(self, plot_each_domain = False):
        labels = ['Attraction',  'Taxi', 'Restaurant','Train', 'Police', 'Hotel', 'Hospital']
        
        domains = [i for i in labels if i.lower() in self.recorders]
        infos = [self.recorders[i.lower()].get_info() for i in domains]

        ### tot_num
        # plt.subplot(1, 2, 1)
        plt.figure(figsize=(10, 7), dpi=300)
        font = {'weight' : 'bold','size' : 25}
        plt.title('Frequency of domain', font, pad=30)
        y = [i[0] for i in infos]
        s = sum(y)
        sizes = [i * 100 / s for i in y]

        patches,l_text,p_text = plt.pie(
            sizes, 
            autopct='%1.1f%%', #显示百分比方式
            shadow=False, #阴影效果
            startangle=90,
            labels = domains,
            pctdistance=0.7
        )
        plt.axis('equal')
        for t in l_text:
            t.set_size(25)
        for t in p_text:
            t.set_size(20)
        plt.savefig('results/%s/Frequency_of_domain.png'%self.modelname)
        plt.close()

        plt.figure(figsize=(12, 7), dpi=300)

        font1 = {'weight' : 'normal','size' : 20}

        font2 = {'weight' : 'bold','size' : 22}

        font3 = {'weight' : 'bold','size' : 35}



        x1 = list(range(len(domains)))
        x1 = np.array(x1)
        x2 = x1 + 0.1
        x3 = x2 + 0.1
        x4 = x3 + 0.1
        plt.tick_params(axis='y', labelsize=20)
        plt.tick_params(axis='x', labelsize=22)
        plt.ylabel('score', font2)   
        plt.ylim(0, 1)
        plt.xlabel('domain', font2)
        plt.title('Performance for each domain', font3, pad=16)
        y = [i[1] for i in infos]
        plt.bar(x1, y, width=0.1, align='center', label='Success rate')
        y = [i[2] for i in infos]
        plt.bar(x2, y, width=0.1, align='center', tick_label=domains, label='Precision')
        y = [i[3] for i in infos]
        plt.bar(x3, y, width=0.1, align='center', label='Recall')
        y = [i[4] for i in infos]
        plt.bar(x4, y, width=0.1, align='center', label='Inform F1')
        plt.legend(loc=2,prop=font1)
        plt.savefig('results/%s/Performance_for_each_domain.png'%self.modelname)
        plt.close()

    def plot_freq(self):
        
        labels = ['Attraction',  'Taxi', 'Restaurant','Train', 'Police', 'Hotel', 'Hospital']
        
        domains = [i for i in labels if i.lower() in self.recorders]
        infos = [self.recorders[i.lower()].get_info() for i in domains]

        plt.figure()
        x = list(range(1, len(domains) + 1))
        # ax = plt.subplot(2, 2, 1)
        y = [i[5] for i in infos]

        if sum(y) == 0: return
        
       

        plt.figure(figsize=(10, 7), dpi=300)
        self.plot_pi(y, domains)
        font = {'weight' : 'bold','size' : 25}
        plt.title('Proportions of the dialogue loop', font, pad=30)
        plt.savefig('results/%s/Proportions_of_the_dialogue_loop.png'%self.modelname)
        plt.close()



    def format_result(self):
        infos = [self.recorders[i].get_info() for i in self.recorders]
        domains = [i for i in self.recorders]

        cols = ['Total Num', 'Succ Rate', 'Precision', 'Recall', 'F1', 'Dialog Loop Failed Rate', 'Dialog Turn (Succ)', 'Dialog Turn (All)']
        table = []
        for i in range(len(infos)):
            tmp = [domains[i]]
            tmp.append(infos[i][0])
            tmp += ['%.3f'%t for t in infos[i][1:5]]
            tmp.append('%.3f'%(infos[i][5] / infos[i][0]))
            tmp.append('%.3f'%infos[i][6])
            tmp.append('%.3f'%infos[i][7])
            table.append(tmp)
        
        return cols, table



if __name__ == "__main__":
    # now = time.time()  # 当前时间 float类型
    print(time.strftime("%Y-%m-%d %H:%M:%S"))