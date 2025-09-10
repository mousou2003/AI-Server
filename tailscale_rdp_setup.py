"""
Tailscale + RDP Always-On Setup for Windows
-------------------------------------------
Run this script **as Administrator** on the Windows machine you want to reach.
It will:
  1) Enable Remote Desktop (RDP) and firewall rules
  2) Install & start the Tailscale Windows Service (always on, before login)
  3) Mark the Tailscale network profile as Private
  4) (Optional) Tweak power settings to prevent sleep on AC power
  5) Print your Tailscale IP to use with mstsc.exe

Requirements:
- Windows 10/11
- Tailscale already installed and logged in at least once
- Run from an elevated (Administrator) prompt:  py -3 tailscale_rdp_setup.py
"""

# ================= DATA SECTION: PowerShell scripts =================
PS_SET_TAILSCALE_PRIVATE = (
    '$adapters = Get-NetAdapter | Where-Object {$_.InterfaceDescription -like "*Tailscale*"}\n'
    'if ($adapters) {\n'
    '  $allOk = $true\n'
    '  foreach ($a in $adapters) {\n'
    '    try {\n'
    '      Set-NetConnectionProfile -InterfaceIndex $a.IfIndex -NetworkCategory Private -ErrorAction Stop\n'
    '      Write-Output "Set Tailscale adapter ($($a.Name)) to Private."\n'
    '    } catch {\n'
    '      Write-Output "Failed to set network category for $($a.Name): $($_.Exception.Message)"\n'
    '      $allOk = $false\n'
    '    }\n'
    '  }\n'
    '  if ($allOk) { exit 0 } else { exit 1 }\n'
    '} else {\n'
    '  Write-Output "No Tailscale adapter found yet. It may appear after the service starts."\n'
    '  exit 2\n'
    '}'
)

PS_SET_TAILSCALE_PUBLIC = (
    '$adapters = Get-NetAdapter | Where-Object {$_.InterfaceDescription -like "*Tailscale*"}\n'
    'if ($adapters) {\n'
    '  foreach ($a in $adapters) {\n'
    '    try {\n'
    '      Set-NetConnectionProfile -InterfaceIndex $a.IfIndex -NetworkCategory Public -ErrorAction Stop\n'
    '      Write-Output "Set Tailscale adapter ($($a.Name)) to Public."\n'
    '    } catch {\n'
    '      Write-Output "Failed to set network category for $($a.Name): $($_.Exception.Message)"\n'
    '    }\n'
    '  }\n'
    '} else {\n'
    '  Write-Output "No Tailscale adapter found."\n'
    '}'
)

import ctypes
import os
import shutil
import sys
import time
from utility_manager import UtilityManager

def is_admin() -> bool:
    try:
        return ctypes.windll.shell32.IsUserAnAdmin() != 0
    except Exception:
        return False

def which_tailscale() -> str:
    default = r"C:\Program Files\Tailscale\tailscale.exe"
    if os.path.isfile(default):
        return default
    found = shutil.which("tailscale")
    return found or default

def which_tailscaled() -> str:
    default = r"C:\Program Files\Tailscale\tailscaled.exe"
    if os.path.isfile(default):
        return default
    found = shutil.which("tailscaled")
    return found or default

def main():
    if not is_admin():
        print("ERROR: This script must be run as Administrator. Right-click cmd.exe and choose 'Run as administrator'.")
        sys.exit(1)

    remove_mode = '--remove' in sys.argv
    tailscale = which_tailscale()
    tailscaled = which_tailscaled()

    if remove_mode:
        print("== Removing all Tailscale RDP setup changes ==")
        UtilityManager.run_subprocess(r'reg add "HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server" /v fDenyTSConnections /t REG_DWORD /d 1 /f', check=False)
        UtilityManager.run_subprocess(r'reg add "HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp" /v UserAuthentication /t REG_DWORD /d 0 /f', check=False)
        UtilityManager.run_subprocess('powershell -NoProfile -ExecutionPolicy Bypass -Command "Disable-NetFirewallRule -DisplayGroup ''Remote Desktop''"', check=False)
        UtilityManager.run_subprocess(r'sc stop Tailscale', check=False)
        UtilityManager.run_subprocess(r'sc delete Tailscale', check=False)
        UtilityManager.run_subprocess(r'reg delete "HKLM\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services" /v fAllowRemoteRPC /f', check=False)
        UtilityManager.run_subprocess(f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{PS_SET_TAILSCALE_PUBLIC}"', check=False)
        UtilityManager.run_subprocess(r'powercfg -change -standby-timeout-ac 15', check=False)
        print("\nAll changes have been removed. RDP and Tailscale service are disabled.")
        return

    if not os.path.isfile(tailscale):
        print(f"ERROR: Tailscale not found at '{tailscale}'. Install Tailscale for Windows and log in once, then re-run.")
        sys.exit(2)

    print("== 1) Enabling Remote Desktop (RDP) and firewall rules ==")
    UtilityManager.run_subprocess(r'reg add "HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server" /v fDenyTSConnections /t REG_DWORD /d 0 /f')
    UtilityManager.run_subprocess(r'reg add "HKLM\SYSTEM\CurrentControlSet\Control\Terminal Server\WinStations\RDP-Tcp" /v UserAuthentication /t REG_DWORD /d 1 /f')
    UtilityManager.run_subprocess('powershell -NoProfile -ExecutionPolicy Bypass -Command "Enable-NetFirewallRule -DisplayGroup ''Remote Desktop''"')

    print("\n== 2) Installing & starting Tailscale Windows Service (always-on) ==")
    service_installed = False
    try:
        UtilityManager.run_subprocess(f'"{tailscale}" service install')
        service_installed = True
    except SystemExit:
        if os.path.isfile(tailscaled):
            print("tailscale service install failed; attempting manual service creation with sc.exe ...")
            UtilityManager.run_subprocess(r'sc query Tailscale', check=False)
            UtilityManager.run_subprocess(fr'sc create Tailscale binPath= "{tailscaled} /subsystem windows-service" start= auto', check=False)
            service_installed = True
        else:
            print("ERROR: Could not install Tailscale as a service. 'tailscaled.exe' not found.")
            sys.exit(3)

    if service_installed:
        UtilityManager.run_subprocess(f'"{tailscale}" service start', check=False)
        UtilityManager.run_subprocess(r'sc config Tailscale start= delayed-auto', check=False)
        print("Set Tailscale service to 'Automatic (Delayed Start)'.")

    print("\n== 2b) Allow RDP connections before user login via Group Policy registry tweak ==")
    UtilityManager.run_subprocess(r'reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows NT\Terminal Services" /v fAllowRemoteRPC /t REG_DWORD /d 1 /f', check=False)
    print("Enabled Group Policy to allow RDP before login.")

    print("\n== 3) Marking Tailscale network as Private (with retry)==")
    for attempt in range(3):
        result = UtilityManager.run_subprocess(f'powershell -NoProfile -ExecutionPolicy Bypass -Command "{PS_SET_TAILSCALE_PRIVATE}"', check=False)
        if result.returncode == 0:
            break
        elif attempt < 2:
            print("Retrying Tailscale adapter network category in 5 seconds...")
            time.sleep(5)
        else:
            print("WARNING: Could not set Tailscale adapter to Private after several attempts. RDP may not work unless the adapter is set manually.")

    print("\n== 4) (Optional) Prevent sleep on AC power (so the box stays reachable) ==")
    UtilityManager.run_subprocess(r'powercfg -change -standby-timeout-ac 0', check=False)

    print("\n== 5) Try to accept DNS/routes and show your Tailscale IP ==")
    UtilityManager.run_subprocess(f'"{tailscale}" set --accept-dns=true', check=False)
    UtilityManager.run_subprocess(f'"{tailscale}" up', check=False)
    ipv4 = UtilityManager.run_subprocess(f'"{tailscale}" ip -4', check=False).stdout.strip()
    ipv6 = UtilityManager.run_subprocess(f'"{tailscale}" ip -6', check=False).stdout.strip()

    print("\n=== Summary ===")
    UtilityManager.run_subprocess(r'sc query Tailscale', check=False)
    if ipv4:
        print(f"Tailscale IPv4: {ipv4}")
    if ipv6:
        print(f"Tailscale IPv6: {ipv6}")
    print("\nUse mstsc.exe to connect to the Tailscale IP/hostname. You should reach the Windows login screen even after a reboot.")
    print("If the machine still sleeps on battery, consider:  powercfg -change -standby-timeout-dc 0")

if __name__ == "__main__":
    main()