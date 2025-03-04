function setupfolder(folder_path::String)
    if !isdir(folder_path)
      mkdir(folder_path)
    else
      rm(folder_path,recursive=true)
      mkdir(folder_path)
    end
  end