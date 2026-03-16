-- Setup the extension.
local ext = get_current_extension_info()

project_ext(ext)

-- Link folders that should be packaged with the extension.
repo_build.prebuild_link {
    { "data", ext.target_dir.."/data" },
    { "docs", ext.target_dir.."/docs" },
    { "python/impl", ext.target_dir.."/isaacsim/physics/newton/impl" },
    { "python/impl/tensors", ext.target_dir.."/isaacsim/physics/newton/tensors" },  -- Link tensors directly to maintain module structure
    { "python/tests", ext.target_dir.."/isaacsim/physics/newton/tests" },
}

-- Copy the main __init__.py to maintain the module root
repo_build.prebuild_copy {
    { "python/__init__.py", ext.target_dir.."/isaacsim/physics/newton/__init__.py" },
}

