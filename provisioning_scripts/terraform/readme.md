
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
    participant remotedocker as Docker
    %% participant remotestartup as Bash<br/>Session
    actor remotemanual as Remote<br/>Manual

    remotemanual->>+remotemachine: `inception_script_manual.bash`

    remotemachine->>remotemachine: `ghcli-install.bash`<br/>`ghcli-login.bash`<br/>`refresh_ssh_agent.env`
    %% remotemachine->>remotemachine: `ghcli-login.bash`
    %% remotemachine->>remotemachine: `refresh_ssh_agent.env`

    remotemachine->>+remotedocker: `docker`
    remotedocker-->>-remotemachine: `exit`

    remotemachine-->>-remotemanual: (ready)

    remotemanual->>remotemachine:patch_bashrc.sh
    remotemanual->>remotemachine:...

    end

    localuser->>localtf: `UP tfdestroy`
    localtf-->>remotemachine: destroy

    %% end

    Note over remotemachine,remotemanual: Remote Machine (GPU)



```

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


