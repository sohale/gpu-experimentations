How to configure


```
pspace login
pspace machine list
```






First distinguish three ways:
* paperspace CLI
* terraform CLI with "Paperspace Provider"
* (paperspace) API

In fact, there are more:
* paperspace CLI: `pspace`
* `doctl` CLI
* terraform CLI with "Paperspace Provider"
* terraform CLI with "DigitalOcean Provider" (see: https://docs.digitalocean.com/reference/terraform/ )
* API (paperspace)
* API (Digital Ocean)
* UI (web): Paperspace
* UI (web): DigitalOcean
* PyDO
* Other DO libraries: https://docs.digitalocean.com/reference/libraries/ ( godo, pydo, dropletkit. digitalocean.cloud, digitalocean.API, DigitalOceanPHP, Levarel DigitalOcean, python-digitalocean, dots (ts))
* Even more: https://docs.digitalocean.com/reference/opensource/


* Paperspace API:
   * https://docs.digitalocean.com/reference/paperspace/pspace/api-reference/
* Paperspace CLI Preview: https://docs.digitalocean.com/reference/paperspace/pspace/
* Core API, CLI, & SDK (examples: CLI, cURL, Node)
   * https://docs.digitalocean.com/reference/paperspace/core/install/
   * https://docs.digitalocean.com/reference/paperspace/core/commands/ *

See:
* https://docs.digitalocean.com/reference/paperspace/pspace/
* https://docs.digitalocean.com/reference/terraform/

* DigitalOcean Bare Metal GPUs
* GPU worker nodes on Kubernetes

## pspace command
`pspace`

What is a "A Paperspace public API Key"?

* Not `project_apikey_secretname` (formerly `apikey_secret_name`). It is only for "Project", which is a feature only in "Gradient" projects, which is outside out interest).
* `api_key`, the long number. Only shown once.


## misc
