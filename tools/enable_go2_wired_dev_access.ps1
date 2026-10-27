#Requires -RunAsAdministrator
[CmdletBinding()]
param(
    [string]$InterfaceAlias = '',
    [string]$InterfaceDescription = 'Realtek PCIe 2.5GbE Family Controller',
    [ValidatePattern('^192\.168\.123\.\d{1,3}$')]
    [string]$PcAddress = '192.168.123.100',
    [ValidatePattern('^192\.168\.123\.\d{1,3}$')]
    [string]$Go2Address = '192.168.123.18',
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
if ($InterfaceAlias) {
    $adapter = Get-NetAdapter -Name $InterfaceAlias -ErrorAction Stop
} else {
    $adapter = Get-NetAdapter | Where-Object { $_.InterfaceDescription -eq $InterfaceDescription } | Select-Object -First 1
    if (-not $adapter) {
        throw "No adapter matched InterfaceDescription '$InterfaceDescription'. Pass -InterfaceAlias explicitly."
    }
}
$addresses = Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 -ErrorAction SilentlyContinue
$managedAddress = $addresses | Where-Object { $_.IPAddress -eq $PcAddress }

if ($Remove) {
    if ($managedAddress) {
        $managedAddress | Remove-NetIPAddress -Confirm:$false
        Write-Output "Removed temporary Go2 wired development address $PcAddress from $InterfaceAlias."
    } else {
        Write-Output "Temporary Go2 wired development address $PcAddress was not present."
    }
    exit 0
}

if (-not $managedAddress) {
    # Do not alter the adapter's existing DHCP/APIPA address, default gateway,
    # DNS, Wi-Fi configuration, or Tailscale routes. This is a second address
    # scoped only to the Go2 controller subnet for maintenance SSH.
    New-NetIPAddress -InterfaceIndex $adapter.ifIndex -IPAddress $PcAddress -PrefixLength 24 `
        -AddressFamily IPv4 -PolicyStore ActiveStore | Out-Null
}

$sshReachable = Test-NetConnection -ComputerName $Go2Address -Port 22 -InformationLevel Quiet
Get-NetIPAddress -InterfaceIndex $adapter.ifIndex -AddressFamily IPv4 |
    Select-Object IPAddress, PrefixLength, AddressState, PolicyStore |
    Format-Table -AutoSize
if ($sshReachable) {
    Write-Output "GO2_WIRED_DEV_ACCESS_OK ssh=$Go2Address`:22"
} else {
    Write-Warning "Address added, but $Go2Address`:22 is not reachable. Check the cable and Go2 Ethernet endpoint."
    exit 1
}
