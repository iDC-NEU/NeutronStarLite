file(REMOVE_RECURSE
  "libcore.pdb"
  "libcore.a"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/core.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
