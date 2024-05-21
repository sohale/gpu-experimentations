# C++ LSP on Vscode

Successfully set up clangd for vscode. !!!!

Now I have a high-quality LSP for C++.

Quality of life`++`

Fantastico ✨✨🌟💫✨🎆✨🎇😫

I did it first here: https://github.com/sohale/ifc2brep-0/tree/main/scripts/lsp.bash
<!--
-rwxrwxr-x  1 ephemssss ephemssss 1.2K May 16 18:41 lsp.bash
-rwxrwxr-x  1 ephemssss ephemssss  933 May 16 18:37 lsp-server.bash
-rwxrwxr-x  1 ephemssss ephemssss   42 May 16 18:30 lsp-client.bash
-->

## Steps:
1. Install `clangd` ( I have it via clang 18)
   * Identify the path for `clangs`, e.g. `/usr/lib/llvm-18/bin/clangd`

2. Install the following two VSCode Extentions:
    1. MS C++ ...
      ```txt
      Version: 1.20.5
      Publisher: Microsoft
      https://marketplace.visualstudio.com/items?itemName=ms-vscode.cpptools
      ```
    2. The `clangd` extention by "LLVM":
      ```txt
      Version: 0.1.28
      Publisher: LLVM https://marketplace.visualstudio.com/items?itemName=llvm-vs-code-extensions.vscode-clangd
      ```
4. Tun `provisioning_scripts/cpp_lsp_support/cpp-lsp-support.bash` to append the necessary bits to your `.vscode/settings.json`. e.g.,

* ```json
  {
    ...
    "C_Cpp.intelliSenseEngine": "disabled",
    "clangd.path": "/usr/lib/llvm-18/bin/clangd",
    "clangd.arguments":
    [
        "-log=verbose",
        "-pretty",
        "--background-index",
        "--compile-commands-dir=/home/ephemssss/novorender/ifc2brep-0/scripts/"
    ],
    "placeholder": null
  }
  ```


3. Generate a `compile_commands.json` based on your build compiler options
    * ```json
      [{
        "directory": "... some project dir ...",
        "command": "/usr/lib/llvm-18/bin/clang++ -std=c++20 -Isomething -DSOMETHING -o something something.cpp something.cpp -LSOMEPATH -lSOMETHING -lSOMETHING",
        "file": "something.cpp"
      }]```

    * Note: Convert the `"command":` to `clang` commandline if necessary. Also, convert to equivalent Linux conmand if it is a windows command. (Use gpt).

    * Note: Convert/Correct the paths if in Docker, Wine, etc

5. Restart vscode by `> refresh window`

6. Verification of clang after vscode restart:
    1. Verify `clangd` in `>clangd ...`

    2. Navigate to a `.cpp` file (in vscode IDE)

    3. check vscode's console -> drop down -> "clangd"

    4. Clear and renavigate to the aforementioned `.cpp` file

    5. You should see healthy output and no "error" in that "clangd" console output logs

* Each time:
  * Backup each file before and after change.
  * Repeat those restarts until they work.
  * The script `provisioning_scripts/cpp_lsp_support/cpp-lsp-support.bash` Can help in json & backup.
  * `todo:` instrucitons to generate `compile_command.json`

### How to generate / update your compile_commands json file
* i.e. `compile_command.json`
* todo

<!-- https://t.me/lecompile/5153 -->
