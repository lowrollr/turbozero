from typing import List
import bottleneck

from collections import OrderedDict
from matplotlib import pyplot as plt
import IPython.display as display
import numpy as np
import logging

class Metric:
    def __init__(self, name, xlabel, ylabel, pl_type: str ='plot', addons={}, maximize=True, alert_on_best=False, proper_name=None, best=None) -> None:
        self.name = name
        self.ts = []
        self.data = []
        self.plot = plt.figure()
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.pl_type = pl_type
        self.addons = addons
        self.maximize = maximize
        if best is None:
            self.best = float('-inf') if maximize else float('inf')
        else:
            self.best = best
        self.alert_on_best = alert_on_best
        self.proper_name = name.capitalize() if proper_name is None else proper_name

    def add_data(self, ts, data):
        self.ts.append(ts)
        self.data.append(data)
        if self.maximize:
            if data > self.best:
                self.best = data
                if self.alert_on_best:
                    logging.info(f'**** NEW BEST {self.proper_name}: {self.best} ****')
        else:
            if data < self.best:
                self.best = data
                if self.alert_on_best:
                    logging.info(f'**** NEW BEST {self.proper_name}: {self.best} ****')

    def reset_fig(self):
        self.plot = plt.figure()

    def clear_data(self):
        self.ts = []
        self.data = []

    def generate_plot(self):
        
        data = self.data
        ts = self.ts
        if data:
            self.plot.clear()
            ax = self.plot.add_subplot(111)

            if 'window' in self.addons:
                # get data within window
                data = data[-self.addons['window']:]
                ts = ts[-self.addons['window']:]

            if self.pl_type == 'hist':
                ax.hist(data, bins='auto')
            else:
                ax.plot(ts, data)
                ax.annotate('%0.3f' % data[-1], xy=(1, data[-1]), xytext=(8, 0), 
                                        xycoords=('axes fraction', 'data'), textcoords='offset points', color='blue')
            ax.set_title(self.proper_name)
            ax.set_xlabel(self.xlabel)
            ax.set_ylabel(self.ylabel)


            if 'running_mean' in self.addons:
                running_mean = bottleneck.move_mean(data, window=min(self.addons['running_mean'], len(data)), min_count=1)
                ax.plot(ts, running_mean, color='red', linewidth=2)
                ax.annotate('%0.3f' % running_mean[-1], xy=(1, running_mean[-1]), xytext=(8, 0), 
                                        xycoords=('axes fraction', 'data'), textcoords='offset points', color='red')
        display.display(self.plot)

class TrainingMetrics:
    def __init__(self, metrics: List[Metric], eval_metrics: List[Metric]) -> None:
        self.metrics = {
            metric.name: metric for metric in metrics
        }
        self.eval_metrics = {
            metric.name: [metric] for metric in eval_metrics
        }
        self.cur_epoch = 0
        self.cur_episode = 0
    
    def add_episode_data(self, episode_data, print_data = True):
        if print_data:
            logging.info(f'Episode {self.cur_episode}')
            for metric_name, metric_data in episode_data.items():
                logging.info(f'\t{self.metrics[metric_name].proper_name}: {metric_data}')

        for metric_name, metric_data in episode_data.items():
            self.metrics[metric_name].add_data(self.cur_episode, metric_data)

        self.cur_episode += 1

    def add_epoch_data(self, epoch_data):
        for metric_name, metric_data in epoch_data.items():
            self.metrics[metric_name].add_data(self.cur_epoch, metric_data)
        self.cur_epoch += 1
        for k in self.eval_metrics.keys():
            if self.eval_metrics[k]:
                self.eval_metrics[k].append(Metric(k, self.eval_metrics[k][-1].xlabel, self.eval_metrics[k][-1].ylabel, pl_type=self.eval_metrics[k][-1].pl_type, \
                                                   addons=self.eval_metrics[k][-1].addons, maximize=self.eval_metrics[k][-1].maximize, alert_on_best=self.eval_metrics[k][-1].alert_on_best, \
                                                    proper_name=self.eval_metrics[k][-1].proper_name, best=self.eval_metrics[k][-1].best))
    
    def reset_all_figs(self): # for matplotlib compatibility
        for metric in self.metrics.values():
            metric.reset_fig()
        for metric_list in self.eval_metrics.values():
            for metric in metric_list:
                metric.reset_fig()  

    def add_eval_data(self, eval_data):
        for metric_name, metric_data in eval_data.items():
            self.eval_metrics[metric_name][self.cur_epoch].add_data(self.cur_epoch, metric_data)
    
    def generate_plots(self):
        display.clear_output(wait=False)
        for metric in self.metrics.values():
            metric.generate_plot()
        for metrics in self.eval_metrics.values():
            metrics[-1].generate_plot()
