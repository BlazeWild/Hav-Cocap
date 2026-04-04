conda activate ./env


/usr/bin/tmux new -s work
/usr/bin/tmux attach -t work
/usr/bin/tmux kill-session -t work

unset PROMPT_COMMAND
