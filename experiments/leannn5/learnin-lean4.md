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

### Q: Package and folder issues:
Issues about not being in the root: (apparently):
   * "Make sure that the folder you have opened in VS Code is the root of your Lean 4 project, where your lean-toolchain file and, ideally, your lakefile.lean are located."
   * `^` apparently. But I dont want that.

   * What is ".code-workspace"? There was a` old-repo-root/lean.code-workspace`.
      * I see!

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
