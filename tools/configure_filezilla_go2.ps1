[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^go2-[a-f0-9]{6}$')]
    [string[]]$Node,

    [string]$UserName = 'unitree',

    [string]$PrivateKeyFile = "$HOME/.ssh/id_ed25519",

    [string]$MagicDnsSuffix = 'tailb73f8d.ts.net'
)

$ErrorActionPreference = 'Stop'
if (-not (Test-Path -LiteralPath $PrivateKeyFile)) {
    throw "Private key not found: $PrivateKeyFile"
}

$configDirectory = Join-Path $env:APPDATA 'FileZilla'
$siteManager = Join-Path $configDirectory 'sitemanager.xml'
New-Item -ItemType Directory -Force -Path $configDirectory | Out-Null

if (Test-Path -LiteralPath $siteManager) {
    [xml]$document = Get-Content -Raw -LiteralPath $siteManager
} else {
    [xml]$document = '<?xml version="1.0"?><FileZilla3><Servers /></FileZilla3>'
}

$servers = $document.FileZilla3.Servers
$folders = @(
    $servers.SelectNodes('Folder') | Where-Object {
        $_.SelectSingleNode('text()[normalize-space(.)="Go2 Edge Fleet"]')
    }
)
if ($folders.Count -eq 0) {
    $folder = $document.CreateElement('Folder')
    $folder.SetAttribute('expanded', '1')
    $folder.AppendChild($document.CreateTextNode('Go2 Edge Fleet')) | Out-Null
    $servers.AppendChild($folder) | Out-Null
} else {
    $folder = $folders[0]
    foreach ($extraFolder in @($folders | Select-Object -Skip 1)) {
        foreach ($existingServer in @($extraFolder.SelectNodes('Server'))) {
            $folder.AppendChild($existingServer) | Out-Null
        }
        $servers.RemoveChild($extraFolder) | Out-Null
    }
}

function Add-TextElement {
    param(
        [System.Xml.XmlElement]$Parent,
        [string]$Name,
        [string]$Value
    )
    $element = $document.CreateElement($Name)
    $element.InnerText = $Value
    $Parent.AppendChild($element) | Out-Null
}

foreach ($robot in $Node) {
    foreach ($existing in @($folder.SelectNodes('Server')) | Where-Object {
        $_.SelectSingleNode('Name').InnerText -eq $robot
    }) {
        $folder.RemoveChild($existing) | Out-Null
    }

    $server = $document.CreateElement('Server')
    Add-TextElement $server 'Host' "${robot}.${MagicDnsSuffix}"
    Add-TextElement $server 'Port' '22'
    Add-TextElement $server 'Protocol' '1'
    Add-TextElement $server 'User' $UserName
    Add-TextElement $server 'Logontype' '5'
    $keyPath = (Get-Item -LiteralPath $PrivateKeyFile).FullName -replace '\\+', '\'
    Add-TextElement $server 'Keyfile' $keyPath
    Add-TextElement $server 'EncodingType' 'UTF-8'
    Add-TextElement $server 'BypassProxy' '1'
    Add-TextElement $server 'Name' $robot
    Add-TextElement $server 'SyncBrowsing' '0'
    Add-TextElement $server 'DirectoryComparison' '0'
    $folder.AppendChild($server) | Out-Null
}

$settings = [System.Xml.XmlWriterSettings]::new()
$settings.Indent = $true
$settings.IndentChars = "`t"
$settings.NewLineChars = "`r`n"
$settings.Encoding = [System.Text.UTF8Encoding]::new($false)
$writer = [System.Xml.XmlWriter]::Create($siteManager, $settings)
try {
    $document.Save($writer)
} finally {
    $writer.Dispose()
}

Write-Host "Configured FileZilla SFTP sites: $($Node -join ', ')"
