
Steps:

* Note:
   * originally from: experiments/raw1/localmachine-tfinit.bash
   * Then https://github.com/sohale/pocs_for_nikolai/blob/main/terraform/environments/neurotalk/readme.md
       * https://github.com/sohale/pocs_for_nikolai/blob/main/terraform/common/localmachine/up.bash
   * Then again here:
       * Instructions at: provisioning_scripts/terraform/common/localmachine/up.bash

* go to
    * `provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/`
    * e.g.
        * `cd ~/gpu-experimentations/provisioning_scripts/terraform/environments/cuda-ptx-hardcoded-dev-experiments/`
    * where, there is a `main.tf`.



# OUTDATED (to be adapted to this new one  )
### cuda-ptx-hardcoded-dev-experiments
## see instead:
https://github.com/sohale/pocs_for_nikolai/blob/main/terraform/environments/neurotalk/readme.md
# .

The "*manual*" for setting up the `environment` for the NeuroTalk demo.

## Manual steps: registration on chosen Cloud GPU provider
### Paperspace

The following steps need ot be done manually.
Terraform, logically, should not and does not handle these. For that, see the next `##`-level step.

Current "settings": (intention)
  * One team, one only person for now.
  * Notes & scope:
    * will change (more people will be in the team, the ownership may be tansferred, etc)
    * Billing will be separate and documentd.

* Register on paperspace  (once only, done)
    * here: `sohale@gmail.com`
    * store safely: password. (I did not set up 2FA)
    * I linked ot my Google OAuth.
    * extra: Name, bio, linked-acocunts: github
* Set up an SSH:
    * typical URL: https://console.paperspace.com/account/settings/ssh-keys
    * Needs its own process.
    * Store safely (the public key).
    * Causion: Never share private key in here or in  any other circumstance. It needs to stay there only. Noody will even need to read a private key. Apart from  "your". Laptop's sshd. (Your: you = legal entity, individual, per laptop, it is peronal-private, like a toothbrush. The circle that you share your personal passwords with. Don't share in team, org, etc).

        <!-- Provisional: -->
        <!-- (part of: "How" from sctach:, ie this file) -->
        <!-- Verified. Led to connectin using ssh. -->

        1. Create an ssh
        2. add to paperspace
        3. do this tfapply
        4. do the ssh, you should be able to connect.
          * no need for scripts for ssh, etc



* (Not used here): Take note of your user's "ID".
* Used: acocunt user name (ID = email here)
    * Here `sohale@gmail.com`
    * Used in the repo as:
        *`terraform/environments/neurotalk/main.tf : data "paperspace_user" "lead-engineer-user" { email }`
    * exactly once

* As paid (once only, done)
    * hint: "Billing"
    * hint: You will need to naviate to Digital Ocean
        * as of: (first half of 2024)
        * This step is key. I was stuck once.
    * Billing
       * Important (see below)
       * Risk ID: "large cloud bill"

* Navigate to "gradient"
    * as opposed to "Core"
    * This step is key. I was stuck once.
    * mouse: top left

* Create a team (once only, done)
    * here: `neuraspin`
    * take note of the ID: `p9estr9vho9`

* Create a project
    * here: `neurotalk-demo`
    * take note of the ID: `p9estr9vho9`
    * from:  "settings"
    * bby "settings" I mean: Typical URL:  https://console.paperspace.com/neuraspin/projects/p9estr9vho9/settings
    * amend: Same link You can amend: project "Name")

* Now you are ready to make use of terraform. See next.

## Terraform steps
### Manual
* Manual installation
    * Meta:
        * (only once)
        * on "local machine": per "localmachine onboarding"!
    * Run the script:
       * `terraform/common/localmachine/install-terraform-on-localmachine.bash`
       * done on Ubuntu 22 LTS on cloud
    * Info:
        * this will server a sa "localmachine", a concept specific to Terrraform.
           * sometimes, confusingly called `backend` and `;ocal`, to be more precise, a `local backend`.

    * Proceed with the next step: The scripted part is part of this
    <!-- Follow some instrucitons ih this file, that lead to `tfapply` -->

### Scripted part
* There is a life-cycle here
* life cycle:
    * tf init
    * tf plan
    * tf apply
        * (be careful)

    TODOL update above

    * just `tfapply`
      Take  note of the output:
        ```txt
        public_ip_outcome = "74.82.28.165"
        user_debug = {
          ...
        ```
    * Then , (since IP is usually changed)
    `ssh-keygen -f "/home/ephemssss/.ssh/known_hosts" -R "74.82.28.165"`
    * then do the ssh
        ```bash
        #ssh -v paperspace@74.82.28.165
        ssh -v -i ~/.ssh/paperspace_sosi_fromlinux paperspace@74.82.28.165
        ```
        <!-- You will need a password. Why? -->
        You must specify the sshkey name!
        <!-- I finally managed to connect to it using ssh. Sovled the password thing. The solution was this: I needed to specify the keypair name in my `ssh` command. That's it. -->

* Pitfalls:
   * This may happen
       * details: `Error on CreateMachine: Status Code 403, Response Body: {"error":{"message":"Billing/account problem: Account has been flagged. Contact support@paperspace.com."`
       * What to do when this happaned?
          * (TBC)

### One-clicked
* (TBC)

### Cloud cost risks to be mindful
* Make sure you do the terraform destroy after use
* Make sur eyou set Billing thresholds (see step ... above)
       * Risk ID: "large cloud bill"



## Risks & Mitigations to be mindful & practice-ful & disciplined about
* Risks:
   * Risk ID: "large cloud bill"
      * Extent (severity):
          * Can be thousands and millions of dollars
      * Manual check every day, then, once every few days. (unless inactive)
         * It is easy to Remove the VM on Web UI
         * Check even after "automated tf-based" (see below)
      * Automated: Billing thresholds
          * Using "life-cycle"ed-cli
              * Uses Terraform
          * is "tf-based" (via TF)

* Mitigations:
   * Mitigation ID "Billing thresholds"
      * Instrucitons (TBC)
