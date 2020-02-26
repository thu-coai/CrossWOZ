import sys, os


class HTMLWriter():
    def __init__(self, path):
        self.path = path
        self.index = open(path, 'w')
        self.index.write('<html><body>')

    def write_title(self, title):
        self.title = title

        self.index.write('<h1>%s</h1>'%title)

    def write_line(self, content):
        self.index.write('<h3>%s</h3>'%content)
        # self.index.write('<br/>')

    def write_done(self):
        self.index.write('</body></html>')
        self.index.close()

    def write_table(self, cols, data):
        self.index.write('<table width="900" border="1">')
        self.index.write('<tr><th>&nbsp;</th>')
        for col in cols:
            self.index.write('<th>%s</th>'%col)
        self.index.write('</tr>')

        for line in data:
            self.index.write('<tr>')
            for d in line:
                self.index.write('<td align="center">%s</td>'%d)
            self.index.write('</tr>')

        self.index.write('</table>')
        
    def write_nlu_fail(self, dalist, mode='System'):
        self.index.write('<b> %s NLU Failed Dialog Act:</b>'%mode)
        if len(dalist) == 0:
            self.index.write('<p>Nothing</p>')
            return
        self.index.write('<ul>')
        for i in dalist:
            self.index.write('<li>')
            self.index.write(i[0])
            self.index.write('<ul>')
            self.index.write('<li>Occur Num:   %s</li>'%i[1])
            self.index.write('<li>NLU Output</li>')
            self.index.write('<ul>')
            for j in i[2]:
                self.index.write('<li>%s'%j[0])
                self.index.write('&nbsp;')
                self.index.write('&nbsp;')
                self.index.write('&nbsp;')
                self.index.write('&nbsp;')
                self.index.write('Occur Num:    %s'%j[1])
                self.index.write('</li>')
            self.index.write('</ul></ul>')
        self.index.write('</ul>')

    def write_png(self):
        domain_freq_path = 'Frequency_of_domain.png'
        perform_path = 'Performance_for_each_domain.png'
        self.index.write('<table><tr>')

        self.index.write('<td><img src="%s" width=600 border=0/></td>'%domain_freq_path)
        self.index.write('<td><img src="%s" width=600 border=0/></td>'%perform_path)
        self.index.write('</tr></table>')

    def write_dialog_loop_png(self, modelname):
        path = 'Proportions_of_the_dialogue_loop.png'
        
        if os.path.exists('results/%s/%s'%(modelname, path)):
            
            self.write_line('Dialogue Loop')
            self.index.write('<img src="%s" width=800 />'%path)

    def write_list(self, title, dalist):
           
        self.index.write('<b> %s</b>'%title)
        if len(dalist) == 0:
            self.index.write('<p>Nothing</p>')
            return
        self.index.write('<ul>')
        for i in dalist:
            self.index.write('<li>')
            self.index.write(i[0])
            self.index.write('&nbsp;')
            self.index.write('&nbsp;')
            self.index.write('&nbsp;')
            self.index.write('&nbsp;')
            self.index.write('Occur Num:     %s</li>'%i[1])
        self.index.write('</ul>')

    def write_metric(self, suc, pre, rec, f1, turn_suc, turn):
        self.index.write('<p> Success Rate: %.1f %%</p>'%(100*suc))
        self.index.write('<p> (Precision, Recall, F1)   :   (%.3f,  %.3f,  %.3f) </p>'%(pre, rec, f1))
        self.index.write('<p> Average Dialog Turn (Succ): %.3f </p>'%turn_suc)
        self.index.write('<p> Average Dialog Turn (All): %.3f </p>'%turn)


    def write_domain(self, domain, suc, pre, rec, f1, turn_suc, turn, dalist_sys, dalist_usr, cyclist, badlist, rnilist, inrlist):
        self.write_line('Domain %s'%domain)
        self.index.write('<div style="margin-left:50">')
        self.write_line('Overall Results')
        self.write_metric(suc, pre, rec, f1, turn_suc, turn)
        self.write_nlu_fail(dalist_sys, 'System')
        self.write_nlu_fail(dalist_usr, 'User')
        self.write_list('Dialog Loop', cyclist)
        self.write_list('Bad Inform Dialog Act', badlist)
        self.write_list('Request But Not Inform Dialog Act', rnilist)
        self.write_list('Inform But Not Request Dialog Act', inrlist)
        self.index.write('</div>')

    def report_HTML(self, cols, table):
        self.write_line('Metric')
        self.write_table(cols, table)
        
    
if __name__ == "__main__":
    writer = HTMLWriter('index.html')
    writer.write_title('Test Report')
    writer.write_line('Time: %s'%time)
    writer.write_line('Model Name:')
    writer.write_table(['pre', 'rec', 'f1'], [['attr', 123, '2', '3'], ['wd', 2, 4, 6]])
    writer.write_line('Domain Attraction')
    writer.index.write('<div>')
    tmp = ['da1', 0.7, [('fda1', 0.4), ('fda2', 0.5), ('fda3', 0.1)]]
    dalist = [tmp, tmp, tmp]
    cyclist = [('da1', 0.5), ('da1', 0.5), ('da1', 0.5)]
    writer.write_nlu_fail(dalist)
    writer.write_list('Cycle Dialog Act', cyclist)
    writer.write_list('Bad Inform Dialog Act', cyclist)
    writer.write_list('Request But Not Inform Dialog Act', cyclist)
    writer.write_list('Inform But Not Request Dialog Act', cyclist)
    writer.index.write('</div>')
    writer.write_done()