import sys
import optparse
import importlib
import time
import os

import numpy as np

try: import simplejson as json
except ImportError: import json

from collections import OrderedDict

from spearmint.tasks.task_group       import TaskGroup

from spearmint.resources.resource import parse_resources_from_config
from spearmint.resources.resource import print_resources_status

from spearmint.utils.parsing import parse_db_address

def get_options():
    parser = optparse.OptionParser(usage="usage: %prog [options] directory")

    parser.add_option("--config", dest="config_file",
                      help="Configuration file name.",
                      type="string", default="config.json")

    (commandline_kwargs, args) = parser.parse_args()

    # Read in the config file
    expt_dir  = os.path.realpath(os.path.expanduser(args[0]))
    if not os.path.isdir(expt_dir):
        raise Exception("Cannot find directory %s" % expt_dir)
    expt_file = os.path.join(expt_dir, commandline_kwargs.config_file)

    try:
        with open(expt_file, 'r') as f:
            options = json.load(f, object_pairs_hook=OrderedDict)
    except:
        raise Exception("config.json did not load properly. Perhaps a spurious comma?")
    options["config"]  = commandline_kwargs.config_file


    # Set sensible defaults for options
    options['chooser']  = options.get('chooser', 'default_chooser')
    if 'tasks' not in options:
        options['tasks'] = {'main' : {'type' : 'OBJECTIVE', 'likelihood' : options.get('likelihood', 'GAUSSIAN')}}

    if not os.path.exists(expt_dir):
        sys.stderr.write("Cannot find experiment directory '%s'. "
                         "Aborting.\n" % (expt_dir))
        sys.exit(-1)

    return options, expt_dir

def main():
    options, expt_dir = get_options()

    resources = parse_resources_from_config(options)

    # Load up the chooser.
    chooser_module = importlib.import_module('spearmint.choosers.' + options['chooser'])
    chooser = chooser_module.init(options)
    experiment_name     = options.get("experiment-name", 'unnamed-experiment')

    # Get a suggestion for the next job
    task_names = ['main']
    resource_name = ''
    suggested_input = get_suggestion(chooser, task_names, options)
    print 'suggested_input:'
    print suggested_input
    print


# TODO: support decoupling i.e. task_names containing more than one task,
#       and the chooser must choose between them in addition to choosing X
def get_suggestion(chooser, task_names, options):

    if len(task_names) == 0:
        raise Exception("Error: trying to obtain suggestion for 0 tasks ")

    experiment_name = options['experiment-name']

    # We are only interested in the tasks in task_names
    task_options = { task: options["tasks"][task] for task in task_names }
    # For now we aren't doing any multi-task, so the below is simpler
    # task_options = options["tasks"]

    task_options = {'main': options['tasks']['main']}


    task_group = TaskGroup(task_options, options['variables'])

    # Load the tasks from the database -- only those in task_names!
    task_group = load_task_group(task_group, options)

    # Load the model hypers from the database.
    hypers = load_hypers()

    # "Fit" the chooser - give the chooser data and let it fit the model.
    hypers = chooser.fit(task_group, hypers, task_options)

    # Save the hyperparameters to the database.
    save_hypers(hypers)

    # Ask the chooser to actually pick one.
    suggested_input = chooser.suggest()

    jobs = load_jobs(task_group)

    job_id = len(jobs) + 1

    job = {
      'params' : task_group.paramify(suggested_input),
      'status' : 'pending'
    }

    jobs.append(job)
    save_jobs(task_group, jobs)

    return suggested_input

def save_hypers(hypers):
    if hypers:
      fd = open('/users/hengganc/hypers.json', 'w')
      hypers['main']['hypers']['beta_alpha'] = hypers['main']['hypers']['beta_alpha'].tolist()
      hypers['main']['hypers']['ls'] = hypers['main']['hypers']['ls'].tolist()
      hypers['main']['hypers']['beta_beta'] = hypers['main']['hypers']['beta_beta'].tolist()
      json.dump(hypers, fd)

def load_hypers():
    try:
      fd = open('/users/hengganc/hypers.json', 'r')
      hypers = json.load(fd)
      hypers['main']['hypers']['beta_alpha'] = np.array(hypers['main']['hypers']['beta_alpha'])
      hypers['main']['hypers']['ls'] = np.array(hypers['main']['hypers']['ls'])
      hypers['main']['hypers']['beta_beta'] = np.array(hypers['main']['hypers']['beta_beta'])
      return hypers
    except:
      return {}

def load_jobs(task_group):
    jobs = []
    try:
      fd = open('/users/hengganc/jobs.json', 'r')
    except:
      print 'No jobs!'
      return []
    for line in fd:
      strs = line.split()
      if len(strs) < 2:
        break
      inputs = []
      for i in range(2, len(strs)):
        inputs.append(float(strs[i]))
      inputs = np.array(inputs)
      job = {'params' : task_group.paramify(inputs)}
      if strs[0] == 'P':
        job['status'] = 'pending'
      else:
        job['status'] = 'complete'
        job['value'] = float(strs[0])
      jobs.append(job)
    return jobs

def save_jobs(task_group, jobs):
    fd = open('/users/hengganc/jobs.json', 'w')
    for job in jobs:
      line = ''
      if job['status'] == 'pending':
        line = line + 'P P'
      else:
        line = line + ' ' + str(job['value']) + ' 1'
      inputs = task_group.vectorify(job['params'])
      for input in inputs:
        line = line + ' ' + str(input)
      fd.write(line)
      fd.write('\n')

def load_task_group(task_group, options):
    jobs = load_jobs(task_group)

    if jobs:
        task_group.inputs  = np.array([task_group.vectorify(job['params'])
                for job in jobs if job['status'] == 'complete'])

        task_group.pending = np.array([task_group.vectorify(job['params'])
                for job in jobs if job['status'] == 'pending'])

        task_names = ['main']
        task_group.values  = {task : np.array([job['value']
                for job in jobs if job['status'] == 'complete'])
                    for task in task_names}

        task_group.add_nan_task_if_nans()

        # TODO: record costs

    return task_group

if __name__ == '__main__':
    main()
