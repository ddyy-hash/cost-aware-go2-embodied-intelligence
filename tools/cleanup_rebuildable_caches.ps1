[CmdletBinding()]
param()

$ErrorActionPreference = "Stop"
$root = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot "..")).Path.TrimEnd('\')
$cacheNames = @("__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache")
$targets = Get-ChildItem -LiteralPath $root -Recurse -Directory -Force -ErrorAction SilentlyContinue |
    Where-Object { $cacheNames -contains $_.Name } |
    Sort-Object FullName -Descending

$removed = 0
foreach ($target in $targets) {
    $full = (Resolve-Path -LiteralPath $target.FullName).Path.TrimEnd('\')
    if ($full -eq $root -or -not $full.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove a path outside the repository: $full"
    }
    Remove-Item -LiteralPath $full -Recurse -Force
    $removed++
}

Write-Output "REGENERABLE_CACHE_DIRS_REMOVED=$removed"

$duplicatePairs = @(
    @("tmp\cyclonedds-0.10.2.tar.gz", "output\robot-release\sdk-vendor\cyclonedds-0.10.2.tar.gz"),
    @("tmp\typing_extensions-4.7.1-py3-none-any.whl", "output\robot-release\sdk-vendor\typing_extensions-4.7.1-py3-none-any.whl"),
    @("tmp\unitree_sdk2_python-65691c8.tar.gz", "output\robot-release\sdk-vendor\unitree_sdk2_python-65691c8.tar.gz")
)
$duplicatesRemoved = 0
foreach ($pair in $duplicatePairs) {
    $source = (Resolve-Path -LiteralPath (Join-Path $root $pair[0])).Path
    $backup = (Resolve-Path -LiteralPath (Join-Path $root $pair[1])).Path
    if (-not $source.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase) -or
        -not $backup.StartsWith($root + '\', [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to compare a path outside the repository."
    }
    if ((Get-FileHash -Algorithm SHA256 -LiteralPath $source).Hash -ne
        (Get-FileHash -Algorithm SHA256 -LiteralPath $backup).Hash) {
        throw "Refusing to remove a non-identical duplicate: $source"
    }
    Remove-Item -LiteralPath $source -Force
    $duplicatesRemoved++
}
Write-Output "IDENTICAL_VENDOR_COPIES_REMOVED=$duplicatesRemoved"
