import Lake
open Lake DSL

package "linlog1proj" where
  -- add package configuration options here

lean_lib «Linlog1proj» where
  -- add library configuration options here

@[default_target]
lean_exe "linlog1proj" where
  root := `Main
