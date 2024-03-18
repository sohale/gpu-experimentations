# Lean4 Learnings

📓  Only install via `lean4` extention, via the `∀` tha appears afterwards
* Not:
   * Not via github build (takes hours)
   * Not via `apt`
   * Not using the sh / wget command

* Good to know:
   * Must use `elan`. Otherwise , doomed.
   * It assumes lean being in your root folder.
   * It adds many files, including lake/toml etc: (in the repo's root). (add more here).
   * I t installs various things: LPS (language server), etc (add here).

Is this a complete list?
```
./.gitignore
./Leannn5.lean
./Leannn5/Basic.lean
./lake-manifest.json
./lakefile.lean
./lean-toolchain
```

* On Mathlib:
If you dont' have Mathlib, apparently you need to reinstall. `elan self remove` helps.

* Each time you need to `source` something.
   * Apparently' that is, `~/.elan/env`

### Q
What is "a `lean-toolchain` File?
   * Needs to be at the root of your "project directory"

### Q
What is a `.toml` file in general context? Is it not specific to Lean4?

### Q
To use leanpkg.toml or Lake file? which one is more modenn? Why not just stick to `.toml` files?
   * Regarding `toml`:
      * It seems `.toml` is for "Lean 3"? and hence, deprecated?
   * For lake
      * you will need a `lakefile.lean`, "ideally"

### Q: Package and folder issues: (folder & workspae confusion) (in progress)
(in progress)

Issues about not being in the root: (apparently):
   * "Make sure that the folder you have opened in VS Code is the root of your Lean 4 project, where your lean-toolchain file and, ideally, your lakefile.lean are located."
   * `^` apparently. But I dont want that.

   * What is ".code-workspace"? There was a` old-repo-root/lean.code-workspace`.
      * I see!
      * A `.code-workspace` file "is used to define a multi-root workspace" by vscode.
      * Each treated as a separate root by VS Code: `{"folders": [{"path":}, * ]}`
      * The location of the  `*.code-workspace`  file "will affect how relative paths within it are resolved."
      * You may end up opening the (sub-)workspace in vscode. Despite losing github. (Untile later notice. This understanding is in progress)

### Q
The `.vscode/setting` necessary for lean4 extension to work, after restarting:
```json
{
  "lean4.toolchainPath": "/home/ephemssss/.elan/toolchains/leanprover--lean4---v4.7.0-rc2"
}
```
1. That string should match an actual folder name: Use `ls` command:
```txt
/home/ephemssss/.elan/toolchains/leanprover--lean4---v4.7.0-rc2/bin/lean
```
It should match the `elan which lean` command. 📓

2. You unfiortunately cannot use ` :"${home}/... ",` in that json file.


### Make sure
📓 Make sure these two match:
lean4.toolchainPath in `.vscode/setting` with `elan which lean` aommand:

```bash
elan which lean
```

### Q
What are the tools beside `elan`? Lean4 should not be used in vaccume.
* toml, lake, etc etc?

### Tips
* Very handy command: Keep running it.
```bash
elan toolchain list
```
*  Also:
```bash
lean --version
elan which lean
elan toolchain list
```


* Most cli commands start with `elan toolchain` ...

   * You will need to keep doing `elan default` ... .
   * Strangely, it is not `elan toolchain default` ... . Not sure why.

* Mave in mind: Versions are called `toolchain` s.

* (unsure) Be aware of the fake `stable`
```bash
elan toolchain list
```
example output:
```txt
stable (default)
nightly
leanprover/lean4:v4.6.1
```
(on the same topic:)

   * Be awry of `.tmp` there in that list.


  * See: Prefix `leanprover/`, what does it mean?

### Q
Prefix `leanprover/` versus without it, what does it mean?

### tip
The autocomplete, do it this way: (for `bash`). To be used with `source`:

```bash
elan completions bash >that/autocomplete.source
```
then, each time:
```bash
source that/autocomplete.source
```

### Q
Active versus default (versus override)?
`elan toolchain`
`elan show`

```bash
elan show
```
output:
```txt
installed toolchains
--------------------

leanprover/lean4:stable
leanprover/lean4:v4.6.tmp
leanprover/lean4:v4.7.0-rc2 (default)

active toolchain
----------------

leanprover/lean4:v4.7.0-rc2 (overridden by '/home/ephemssss/gpu-experimentations/experiments/leannn5/lean-toolchain')
Lean (version 4.7.0-rc2, x86_64-unknown-linux-gnu, commit 6fce8f7d5cd1, Release)
```

Use:
```lean
#eval Lean.versionString
```


### Q
The difference between Active versus "default" versus "override"?

* default: global
* active: locally withing your project (session or folder?): `lean-toolchain`
* override: is Temporarily

```
leanprover/lean4:v4.7.0-rc2 (overridden by '/home/ephemssss/gpu-experimentations/experiments/leannn5/lean-toolchain')
```

* **Override**: Scope: Local to a folder/directory.
* **Active**: Scope: Local to the current terminal session (or the specific instance of VS Code/extention).
* **Default**: Scope: Global for the current user.

It attributes the "override" to due to the presence of a `lean-toolchain` file.
"This local configuration overrides the global default."

But then, why Override does not use tool-chain (maybe it does)
Answer: Try `elan override`.

Side:
* `lean-toolchain` file when inside a Lean "package"?

* directories can also be assigned their own Lean toolchain manually with `elan override`.

When a directory has an override then any time `lean` or `lake` is run inside that directory, or one of its child directories, the override toolchain will be invoked.


States: active, installed, default, override, (not-installed)

### Actung:
The active/default version of your lean4-extention may be different from the one in your terminal


### Mathlib installtion

When  `Mathlib` does not exist: 'Mathlib.Algebra.Exponential': no such file or directory
* `echo $LEAN_PATH`
* `find ~/.elan -name "Mathlib"`

* Key: You need to add Mathlib as a Dependency in Lake.
* Then you need to build it, when lake file is changed: If no response, you need to clean first. The manifest file may be intruding.
```bash
# elan update
lake clean
lake build
```

The Language server builds the file too. Under the hood. It is invisible. Using "restart file" when it prompts for it:
```txt
Imports of 'Net1.lean' are out of date and must be rebuilt. Restarting the file will rebuild them.
```

### Lake file:

#### Q
Difference between `package  «Leannn5»`  vs `lean_lib  «leannn5»` ?
*  The name in the `lean_lib` declaration should match the directory name


*  There is typically one package declaration per lakefile.lean,

## The language

### f(x)
Did you know `(x)` is invalid?
That's why `f(x)` is invalid, and doe snot mean `((f) (x))`, i.e., `f x`

#### Math
I like to leverage `Mathlib`. So, for my math formulas, I use:
`import Mathlib.Algebra.Exponential`

See Mathlib installtion above, when not found.

### Lambda `(λ`
I'd like to use λ notation for local lambda functions.

Examples:
```lean4

terms_list.foldl (λ sum i, sum + (x ^ i / nat.fact i)) 0.0

```

```lean4
#eval Lean.versionString
```

Cmd+Click on the import. Wow:
`.lake/packages/mathlib/Mathlib/Data/Real/Basic.lean`

.lake/packages/mathlib/docs/overview.yaml
Mentions: Real.exp
but where?

"Simplifier" mentioned in
.lake/packages/mathlib/Mathlib/Analysis/Calculus/Deriv/Basic.lean


Wow:

ℝ -> ℝ

failed to compile definition, consider marking it as 'noncomputable' because it depends on 'Real.instLinearOrderedRingReal', and it does not have executable code
