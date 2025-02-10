
```mermaid
sequenceDiagram


    actor localuser as Manual trigger<br/> (Local Machine)
    participant localtf as TF<br/>(Local M.)

    Note over localuser,localtf: Local Machine

    localuser->>localtf: `UP tfinit`

    rect rgb(240, 247, 255)
    localuser->>+localtf: `UP tfapply`

    create participant remotemachine as Remote
    %% remotemachine: remotemachine, during provisioning
    localtf-->>remotemachine: (provi<br/>sion)
    %% #(create)
    localtf-->>-localuser: (done)

    %% startup (remote)
    end

    %% autonumber

    rect rgb(254, 240, 255)
    %% "create" did not work
    %% participant remotedocker as Docker
    %% participant remotestartup as Bash<br/>Session
    actor remotemanual as Remote<br/>Manual


    %% out of nowhere (1)
    localuser->>+localtf: `UP show_outputs`
    %% localtf->>remotemachine: `upload_scripts_to_there.bash`<br/> (copies inception_script_manual.bash )
    localtf->>localtf: `upload_scripts_to_there.bash`
    localtf->>remotemachine: Copies: `inception_script_manual.bash`
    localtf->>localtf: Runs: `local_manual__setup_at_creation.bash`
    %% remotemachine->>remotemachine: runs: local_manual__setup_at_creation.bash
    localtf->>remotemachine: Copies: `inception_script_manual.bash`
    %% runs locally:
    %%   local_manual__setup_at_creation.bash
    %% uploads scripts:
    %%   inception_script_manual.bash
    %%   (not all?)
    %% the main.tf copies:
    %%    inception_script.tf-template.bash
    %% not!!:
    %% localtf--x-localuser: (ready)
    localtf->>+remotemachine: ssh session (interactive)
    remotemachine-->remotemanual: give the control(!)
    remotemachine--x-localtf: exit
    %%      ssh session end
    localtf--x-localuser: (done)



    remotemanual->>remotemachine: `dot_bashrc.bash`,`patch_bashrc.sh`
    %% the `refresh_ssh_agent.env` is applied as part of `dot_bashrc.bash`, but also 
    %% remotemanual->>remotemachine: patch_bashrc.sh
    %% `patch_bashrc.sh` is not for this. But can be a good extra `.bashrc`. It shall run only once.

    %% can be done later or earlier
    %% remotemanual->>remotemachine: `system_hardware_spec_info.bash`<br/>...


    remotemanual->>remotemachine: inception_script.tf-template.bash`
    %% remotemachine-->-remotemanual: (done)

    remotemanual->>+remotemachine: `inception_script_manual.bash`

    remotemachine->>remotemachine: `ghcli-install.bash`, `ghcli-login.bash` <br/> `refresh_ssh_agent.env` (as part of `inception_script_manual`)<br/> `system_hardware_spec_info`
    %% the `refresh_ssh_agent.env` is applied also at `dot_bashrc.bash`
    %% remotemachine->>remotemachine: `ghcli-login.bash`
    %% remotemachine->>remotemachine: `refresh_ssh_agent.env`

    remotemachine-->>-remotemanual: (ready)

    remotemanual->>+remotemachine: `docker ...`

    create participant remotedocker as Docker
    remotemachine->>remotedocker: `docker`

    remotemanual->>remotedocker: `exit`

    destroy remotedocker
    remotedocker--xremotemachine: `exit`
    remotemachine--x-remotemanual: .


    remotemanual->>remotemachine: `system_hardware_spec_info.bash`, ...

    end


    %% out of nowhere (2)
    localuser->>localtf: `UP ssh_into` <br/> `UP other ...` , `UP bash`


    localuser->>localtf: `UP tfdestroy`
    localtf-->>remotemachine: destroy

    %% end

    Note over remotemachine,remotemanual: Remote Machine (GPU)

    destroy remotemachine
    remotemachine--xlocaltf: .
    localuser->>localtf: `UP tfplan` , `UP tfvalidate`


```
<!-- https://mermaid.js.org/syntax/sequenceDiagram.html -->

## Steps
(See pocs_for_nikolai: `/README.md` )

Three tf stages: `bash ./terraform/common/localmachine/up.bash  SUBCOMMAND`

, where SUBCOMMAND is:
  `tfinit`
  `tfplan`
  `tfapply`
  `show_outputs`


The `show_outputs` can be in future split into three parts: (and also renamed.).
The Three post-tf scripts: (todo):
1. "show output"
2. go there (idepmpotent)
3. do there (light: `ssh` only)

Then inside there, you need to:
* `bash /home/paperspace/scripts-sosi/scripts_to_push/inception_script_manual.bash`

The mini env (source) scripts: in folder: `/home/paperspace/scripts-sosi/`
*  `refresh_ssh_agent.env`    # super minimal
*  `env_context.bash`         # semi-minimal
*  `scripts_to_push/dot_bashrc.bash`   # ok, but dont make it complicated ( ** I think this is no longer used)


Then! (Currenlty auotmatically done as the last part of `inception_script_manual.bash`)
* git clone
* docker run

## Scripting
* separate translatent helper scripts
* some are run as part of `inception_script_manual.bash`, etc
* Two main ones:
* inception_script.tf-template.bash
* ...
* The `show_output` actually runs the ssh session?! and does the actual update?
    * It shall change name

## Internals
### Scripts...
* The `inception_script.tf-template.bash` has two verisons, same name! But on differnt computers? So they won't be confused? (todo: check)

## Precedence and Forking Lineage:

see https://github.com/sohale/pocs_for_nikolai/tree/main/terraform




see ....



Key: see
/myvol/pocs_for_nikolai/README.md

see:
provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/readme.md

e.g.
~/gpu-experimentations/provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/readme.md

?

../../../environment_boxes/neurotalk/scripts_to_push/inception_script.tf-template.bash

For scripts to run inside the created machine:

https://github.com/sohale/pocs_for_nikolai/blob/main/environment_boxes/neurotalk/local_manual__setup_at_creation.bash
https://github.com/sohale/pocs_for_nikolai/tree/main/environment_boxes/neurotalk/scripts_to_push


