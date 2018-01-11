#!/bin/bash
# Helper to automatically attach to TMUX on ssh
# See
# https://stackoverflow.com/questions/7114990/pseudo-terminal-will-not-be-allocated-because-stdin-is-not-a-terminal
# https://stackoverflow.com/questions/1376016/python-subprocess-with-heredocs
export cmd="ssh -t -i $1 -o StrictHostKeyChecking=no $2@$3 tmux a"
echo $cmd
$cmd
