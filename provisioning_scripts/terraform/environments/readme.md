
.
├── common
│   ├── localmachine
│   │   ├── install-terraform-on-localmachine.bash
│   │   ├── up-matmul-cuda-experiment.bash  # The main "UP" script: To up a server.
│   │   └── upload_scripts_to_there.bash    # Called by "up". Run on (from) Local Machine. Triggered by user (should be automated), via up-....bash (`show_outputs` mode). (un-automated: deferred to manual trigger, but via Local Machine)

│   ├── multicloud
│   │   ├── cross-cloud-notes.txt
│   │   ├── list-2.txt
│   │   └── list.last.txt
│   │
│   └── per-provider
│       └── paperspace                           # utilities for each provider? Used to contain setup_anew_ssh.bash, but now moved. ./common/per-provider/paperspace/setup_anew_ssh.bash
│           └── setup_anew_ssh.bash              # Manual utility for creating key pair for `ssh` and github access on the local machine. Example: `paperspace_sosi_fromlinux`
│
├── environments
│   ├── cuda-ptx-hardcoded-dev-experiments
│   │   ├── environment-box
│   │   │   ├── local_manual__setup_at_creation.bash  # Run automatically (via upload_scripts_to_there.bash , called by up). aka SCRIPT1
│   │   │   ├── readme.md
│   │   │   └── scripts_to_push
│   │   │       ├── dot_bashrc.bash           # Remote Machine. manually (later: Shell-Time). Picks the generated "refresh_ssh_agent.env" (removed. was: environments/cuda-ptx-hardcoded-dev-experiments/environment-box/scripts_to_push/dot_bashrc.bash )
     The `refresh_ssh_agent.env` (Which is creaated in ..., now has to be executed manually)
     But also shard usage by `inception_script_manual.bash`,

     Dev. note: which is ...
     inception_script_manual.bash"
  remote name: SCRIPT_FILE_REMOTE_PREDICTED_NAME="$TARGET_SCRIPTS_DIR/inception_script_manual.bash"
  but I see another name: my_scripts_put_here_via_scp (but it is skipped)

Simply useful utilities. Are not orchestrated! (In fact, everything is like that! So, why run the docker everytime in the `inception_script_manual.bash`? A new model/concept of "inception" is needed. Currently, these are helper/utilities/fragments.)


I must extract out Docker from inception_script_manual.bash, since, it was probebly jsut  aest. Everything is extermly separated, for transparency (excpt for github* scripts)



│   │   │       ├── ghcli-install.bash        # Remote Machine.  Installation-Time (machine provisioning).  Picks the generated.
│   │   │       ├── ghcli-login.bash    # run by inception_script_manual.bash on Remote Machine. Triggerd manually (since ht elatter is so).
│   │   │       ├── inception_script.tf-template.bash   # I think it "installs" webserver, etc.
│   │   │       ├── inception_script_manual.bash  # copied to Remote, and shall be run manually in RM terminal.
│   │   │       └── system_hardware_spec_info.bash
│   │   ├── main.tf       # Run via the up-.... . It copies ... (scripts: )  and runs ... (scripts: ) .
│   │   ├── onoff_switch.tf
│   │   ├── outputs.tf
│   │   ├── providers.tf
│   │   ├── readme.md
│   │   ├── terraform.tfstate  # The "terraform cli"s -backend-config="$TF_STATE_FILE"
│   │   ├── terraform.tfstate.backup
│   │   ├── runtime
│   │   │   └── terraform_state
│   │   │       └── plan_delta.dfplan
│   │   ├── tf-temp-runtime
│   │   │   ├── FNAMEB.txt
│   │   │   ├── machine_A5000-twnlo3zj.env
│   │   │   ├── machine_C1-t04azgph.env
│   │   │   └── main-changed.tf.old.txt
│   │   ├── tfvarconfig
│   │   │   ├── cuda-setup-1.tfvars
│   │   │   ├── current_active.tfvars -> entomind.tfvars
│   │   │   ├── current_active_secrets.tfvars -> secrets_entomaind.tfvars
│   │   │   ├── entomind----was--notes.tfvars
│   │   │   ├── entomind.tfvars
│   │   │   ├── ghcli-token.txt
│   │   │   ├── readme-how-to-configure.md
│   │   │   ├── secrets-1.tfvars
│   │   │   └── secrets_entomaind.tfvars
│   │   └── variables.tf
│   ├── nother
│   └── readme.md
├── historical-states
│   ├── list2.txt
│   ├── terraform--17-june-2024.tfstate.historical.json
│   ├── terraform.tfstate.backup
│   └── terraform.tfstate.june18-2024-1126.backup
├── my_providers
│   └── paperspace_instance_control
│       └── paperspace_instance_control.go
└── readme.md

17 directories, 52 files


%% `patch_bashrc.sh` is not for this. But can be a good extra `.bashrc`. It shall run only once.


│   │   ├── inception_script.tf-template.bash -> ....
│   │   ├── machine_.env
