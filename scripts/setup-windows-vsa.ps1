param(
    [string]$Prefix = (Join-Path $env:LOCALAPPDATA 'ImaginationStudio')
)
$ErrorActionPreference = 'Stop'
$commit = '323a0a5f7c636cc70f95464145732f81f04a72e9'
$nodes = Join-Path $Prefix 'ComfyUI\custom_nodes'
$target = Join-Path $nodes 'ComfyUI-Ref2VA-VSA'
if (-not (Test-Path (Join-Path $Prefix 'ComfyUI\main.py'))) {
    throw 'Install the native CUDA/ComfyUI core first. This script does not build H3 or download model weights.'
}
if (Test-Path $target) {
    $revision = git -C $target rev-parse HEAD
    if ($LASTEXITCODE -ne 0 -or $revision -ne $commit) {
        throw 'The existing VSA node is not at the pinned revision; it was not overwritten.'
    }
    $dirty = git -C $target status --porcelain --untracked-files=no
    if ($LASTEXITCODE -ne 0 -or $dirty) {
        throw 'The existing VSA node has changes; it was not overwritten.'
    }
} else {
    git clone --no-checkout https://github.com/Kablex/ComfyUI-Ref2VA-VSA.git $target
    if ($LASTEXITCODE -ne 0) { throw 'VSA node clone failed.' }
    git -C $target -c core.autocrlf=false checkout --detach $commit
    if ($LASTEXITCODE -ne 0) { throw 'Pinned VSA node checkout failed.' }
}
Write-Host 'Pinned experimental VSA node installed; its Apache-2.0 LICENSE is retained.'
Write-Host 'Uses existing Torch and safetensors; no packages, model weights or H3 compiler/toolkit were installed.'
Write-Host 'MiniMax H3 model licences still apply. Restart the Studio, then explicitly run the 22-frame VSA check.'
