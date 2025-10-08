"""
Make sure the specified folder exists.

# Examples
    setupfolder("output")                    # Remove everything inside "output"
    setupfolder("output", ".*")              # Remove everything inside "output"
    setupfolder("output", "all")             # Remove everything inside "output"
    setupfolder("results", ".vtu")           # Remove only .vtu files
    setupfolder("results", [".vtu", ".pvd"]) # Remove multiple file types
    setupfolder("data", nothing)             # Keep existing contents
    setupfolder("data", remove=nothing)      # Keep existing contents
"""
function setupfolder(path::String; remove::Any="all")
  if isdir(path)
    cleandir(path, remove)
  end
  mkpath(path)
end

function cleandir(path::String, remove::String)
  if remove === "all"
    rm(path,recursive=true)
  else
    foreach(rm, filter(endswith(remove), readdir(path,join=true)))
  end
end

function cleandir(path::String, remove::AbstractVector{String})
  foreach(r -> cleandir(path,r), remove)
end

function cleandir(::String, ::Nothing)
end
