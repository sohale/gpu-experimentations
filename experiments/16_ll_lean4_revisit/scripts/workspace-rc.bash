echo "doin it: scripts/workspace-rc.bash"
: -- IGNORE --- '
ls -alth $HOME/.bashrc
if [ -f $HOME/.bashrc ]; then
  echo "Sourcing $HOME/.bashrc :"
  # WHy it does not run this: Because of '$-'
  source "$HOME/.bashrc"

  echo "Sourced $HOME/.bashrc . (Or did it?)"
else
  echo "$HOME/.bashrc not found"
fi

script_dir="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
echo "rc: script_dir=$script_dir"

# No, it's the other way around:
#  . "${script_dir}/workspace-env.bash"
echo "Bye from workspace-rc.bash."
'
