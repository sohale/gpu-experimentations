
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

    rect rgb(254, 240, 255)
    %% "create" did not work
    %% participant remotedocker as Docker
    %% participant remotestartup as Bash<br/>Session
    actor remotemanual as Remote<br/>Manual


    remotemanual->>remotemachine: `dot_bashrc.bash`,`patch_bashrc.sh`
    %% `patch_bashrc.sh` is not for this. But can be a good extra `.bashrc`. It shall run only once.
    %% remotemanual->>remotemachine:patch_bashrc.sh

    remotemanual->>remotemachine: `system_hardware_spec_info.bash`<br/>...

    remotemanual->>remotemachine: inception_script.tf-template.bash`
    %% remotemachine-->-remotemanual: (done)

    remotemanual->>+remotemachine: `inception_script_manual.bash`

    remotemachine->>remotemachine: `ghcli-install.bash`<br/>`ghcli-login.bash`<br/> `refresh_ssh_agent.env`
    %% remotemachine->>remotemachine: `ghcli-login.bash`
    %% remotemachine->>remotemachine: `refresh_ssh_agent.env`

    create participant remotedocker as Docker
    remotemachine->>remotedocker: `docker`

    destroy remotedocker
    remotedocker--xremotemachine: `exit`

    remotemachine-->>-remotemanual: (ready)

    remotemanual->>remotemachine:...

    end

    localuser->>localtf: `UP tfdestroy`
    localtf-->>remotemachine: destroy

    %% end

    Note over remotemachine,remotemanual: Remote Machine (GPU)

    destroy remotemachine
    remotemachine--xlocaltf: .
    localuser->>localtf: `UP tfinit`


```


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


