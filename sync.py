#!/usr/bin/env python
# Sync utility, forked from original by gdb@openai
import argparse
import fcntl
import logging
import os
import re
import select
import subprocess
import sys
import util as u
# In modules, use `logger = logging.getLogger(__name__)`

parser = argparse.ArgumentParser(description='sync')
parser.add_argument('-v', '--verbose', action='count', dest='verbosity',
                    default=0, help='Set verbosity.')
parser.add_argument('--remote', type=str, default="asdf")
parser.add_argument('--name', type=str, default='', help="name of instance to sync with")
args = parser.parse_args()

logger = logging.getLogger()
logger.addHandler(logging.StreamHandler(sys.stderr))
if args.verbosity == 0:
    logger.setLevel(logging.INFO)
elif args.verbosity >= 1:
    logger.setLevel(logging.DEBUG)

class Error(Exception):
    pass
class Resyncd(object):
    def __init__(self, remote, sync):
        self.remote = remote
        self.sync = sync
        self.counter = 0
    def run(self):
        self.resync()
        sources = [sync.source for sync in self.sync]
        fswatch = subprocess.Popen(['fswatch'] + sources, stdout=subprocess.PIPE)
        fl = fcntl.fcntl(fswatch.stdout.fileno(), fcntl.F_GETFL)
        fcntl.fcntl(fswatch.stdout.fileno(), fcntl.F_SETFL, fl | os.O_NONBLOCK)
        while True:
          r, _, _ = select.select([fswatch.stdout], [], [])
          fswatch_output = r[0].read()
          output = fswatch_output.decode('ascii')
          files = output.strip().split("\n") 
          print(files)
          # Ignore emacs swap files
          files = [f for f in files if not re.search("^[.#]", os.path.basename(f))]
          files = set(files)  # remove duplicates from fswatch_output
          if not files:
            continue
          
          print("---")
          print(files)
          print("---")
          self.resync()
          
    def resync(self):
        procs = []
        for sync in self.sync:
            instance = u.get_instances(args.name, verbose=False)[0]
            print("Syncing with ", u.get_name(instance))
            
            command = sync.command(instance)
            popen = subprocess.Popen(command)
            procs.append({
                'popen': popen,
                'command': command,
            })
        # Wait
        for proc in procs:
            print(proc["command"])
            proc['popen'].communicate()
        for proc in procs:
            if proc['popen'].returncode != 0:
                raise Error('Bad returncode from %s: %d', proc['command'], proc['popen'].returncode)
        logger.info('Resync %d complete', self.counter)
        self.counter += 1


class Sync(object):
  # todo: exclude .#sync.py
    excludes = ['*.model', '*.cache', '.picklecache', '.git', '*.pyc', '*.gz']
    def __init__(self, source, dest, modify_window=True, copy_links=False, excludes=[]):
        self.source = os.path.expanduser(source)
        self.dest = dest
        self.modify_window = modify_window
        self.copy_links = copy_links
        self.excludes = self.excludes + excludes
    def command(self, instance, pem_location=''):
        excludes = []
        for exclude in self.excludes:
            excludes += ['--exclude', exclude]

        # todo, rename no_strict_checking to ssh_command

        keypair_fn = u.get_keypair_fn(instance)
        username = u.get_username(instance)
        ip = instance.public_ip_address

        ssh_command = "ssh -i %s -o StrictHostKeyChecking=no"%(keypair_fn,)
        no_strict_checking = ['-arvce', ssh_command]

        command = ['rsync'] + no_strict_checking + excludes
        if self.modify_window:
            command += ['--update', '--modify-window=600']
        if self.copy_links:
            command += ['-L']
        command += ['-rv', self.source, username+"@"+ip + ':' + self.dest]
        print("Running ")
        print(command)
        return command


def main():
    sync = [Sync(source='.', dest='.', copy_links=False),]

    # obtain ssh 
    resyncd = Resyncd(args.remote, sync)
    
    resyncd.run()
    return 0

if __name__ == '__main__':
    sys.exit(main())
