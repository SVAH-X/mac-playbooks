---
slug: tailscale
title: "Tailscale on macOS"
time: "5 min"
color: green
desc: "Secure mesh networking for remote access"
tags: [tools, networking]
spark: "Tailscale"
category: tools
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

Tailscale creates a peer-to-peer mesh VPN using the WireGuard protocol. Unlike traditional VPNs that route all traffic through a central server, Tailscale devices connect directly to each other using NAT traversal — the same technology that lets two people in different countries video-call without either side configuring their router. When direct connection isn't possible (e.g., strict firewalls), Tailscale falls back to encrypted relay servers (DERP servers) automatically.

Each device gets a stable `100.x.x.x` IP address and a DNS hostname via MagicDNS (e.g., `my-mac.tail1234.ts.net`). These addresses never change even when you switch networks. You access your Mac at `my-mac` from your phone, laptop, or any other Tailscale device — from anywhere in the world, with no firewall rules or port-forwarding configured.

The free tier supports 100 devices.

## What you'll accomplish

A Tailscale network (called a "tailnet") connecting your Mac to your other devices. You'll be able to SSH into your Mac from anywhere using its stable Tailscale hostname, and expose your local Ollama API securely to your other Tailscale devices without opening any firewall ports.

## What to know before starting

- **WireGuard**: A modern VPN protocol — faster than OpenVPN, uses a smaller codebase (4,000 lines vs 400,000 for OpenVPN), and is built into the Linux kernel and macOS Network Extension framework. Tailscale uses WireGuard for the actual encryption.
- **NAT traversal**: Your home router uses NAT to let multiple devices share one public IP. Tailscale "punches holes" through NAT using STUN — it gets both devices to contact a coordination server simultaneously, establishing a direct path. This works without any router configuration.
- **MagicDNS**: Tailscale's built-in DNS that maps device names to their Tailscale IPs. When you SSH to `my-mac`, Tailscale resolves this to `100.x.x.x` automatically — no `/etc/hosts` entries needed.
- **Exit nodes**: A Tailscale device configured as an exit node routes all internet traffic from other devices through itself. Useful for privacy or accessing geo-restricted content. Note: this routes ALL your traffic, not just traffic to the exit node device.
- **ACLs**: Access control lists in the Tailscale admin console define which devices can communicate with which other devices on your tailnet. By default, all devices on your tailnet can reach each other.

## Prerequisites

- macOS 10.13+
- A Tailscale account (free tier at tailscale.com — sign up with Google, Microsoft, or GitHub)
- Internet connection during setup (not needed during use once connected)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None — uninstalling Tailscale removes the network extension completely and all changes are reversed

<!-- tab: Install -->
## Step 1: Install via Homebrew or the Mac App

Two installation options with different tradeoffs: Homebrew installs the CLI (`tailscale`) and daemon (`tailscaled`) as background services — better for servers or users who prefer CLI-only. The Mac App (from the App Store or direct download) adds a menu bar icon for easier status checking and connecting/disconnecting.

```bash
# Option A: Homebrew CLI (preferred for servers and power users)
brew install tailscale

# Start the background daemon
brew services start tailscale

# Option B: Mac App (preferred for desktop use)
# Download from: https://tailscale.com/download/mac
# Or: App Store → search "Tailscale"
```

Verify the CLI is available after install:

```bash
tailscale version
# Expected: 1.xx.x
```

## Step 2: Authenticate your Mac

`tailscale up` opens your browser to the Tailscale authentication page. You'll log in with Google, Microsoft, or GitHub — whichever account you used when creating your Tailscale account. After authenticating, your Mac appears in the Tailscale admin console at admin.tailscale.com.

```bash
tailscale up
# Opens browser → log in → Mac appears in admin console
```

After authentication, your terminal shows:

```
Success.
```

Tailscale is now running as a background service. It will automatically reconnect when you restart your Mac.

## Step 3: Verify connectivity

Check which devices are on your tailnet and verify your Mac is connected. The `tailscale ping` command tests direct connectivity to another device — a response under 10ms means a direct WireGuard connection (no relay). A response over 50ms may mean a relay is being used.

```bash
# Show all devices on your tailnet and their status
tailscale status
# Expected output: a table with device names, IPs, and "active" or "offline" status

# Get your Mac's Tailscale IP address
tailscale ip -4
# Expected: 100.x.x.x

# Ping another device on your tailnet by hostname
tailscale ping your-phone
# Expected: pong from your-phone (100.x.x.x) via DERP(nyc) in 45ms
# Direct connection: pong from your-phone (100.x.x.x) in 8ms
```

If `tailscale status` shows no other devices, they either aren't connected to Tailscale or aren't logged into the same account.

<!-- tab: Connect -->
## Step 1: Find your Mac's Tailscale address

Your Mac gets both an IP address (100.x.x.x) and a DNS hostname. The hostname is more convenient because it doesn't change even if you reinstall Tailscale. The `100.x.x.x` range is CGNAT address space reserved specifically for Tailscale — these IPs are private to your tailnet and not routable on the public internet.

```bash
# Get your Tailscale IP
tailscale ip -4
# Example output: 100.68.42.7

# Get your MagicDNS hostname
tailscale status --self
# Shows something like: my-mac (100.68.42.7) linux; tailscale 1.xx
# MagicDNS hostname: my-mac.tail1234.ts.net
```

You can also find both in the Tailscale menu bar app or at admin.tailscale.com → Machines.

## Step 2: SSH into your Mac from another device

SSH must be enabled on the Mac before you can connect (it's off by default in macOS). Enable it in System Settings, then connect from any Tailscale-connected device anywhere in the world.

Enable SSH on your Mac:
```
System Settings → General → Sharing → Remote Login → toggle ON
Optionally: restrict to "Only these users" and add your account
```

Then from any other Tailscale device:

```bash
# Connect using MagicDNS hostname (most convenient)
ssh your-username@my-mac

# Connect using Tailscale IP (works if MagicDNS isn't resolving)
ssh your-username@100.68.42.7

# With a specific identity file
ssh -i ~/.ssh/id_ed25519 your-username@my-mac
```

The first connection asks you to accept the host fingerprint — verify it matches what `ssh-keygen -l -f /etc/ssh/ssh_host_ed25519_key.pub` shows on the Mac.

## Step 3: Advertise as an exit node (optional)

An exit node routes all internet traffic from other devices through your Mac. This is useful for making your laptop appear to be at home when you're traveling, or for routing traffic through a specific network. It requires explicit setup on both the advertising device and the using device.

```bash
# On your Mac: advertise as an exit node
sudo tailscale up --advertise-exit-node

# Approve in the admin console: admin.tailscale.com → Machines → your-mac → Edit route settings → approve exit node

# On another device: route all traffic through your Mac
tailscale up --exit-node=my-mac

# To stop using exit node:
tailscale up --exit-node=
```

<!-- tab: Usage -->
## Step 1: Expose Ollama over Tailscale

By default, Ollama binds to `localhost:11434` — only accessible from the same machine. Setting `OLLAMA_HOST=0.0.0.0` makes it listen on all interfaces, including the Tailscale interface. Any device on your tailnet can then reach your Ollama API.

Security note: all devices on your tailnet can access this. If you share your tailnet with others, use ACLs in the admin console to restrict Ollama access to specific devices.

```bash
# Start Ollama bound to all interfaces (including Tailscale)
OLLAMA_HOST=0.0.0.0:11434 ollama serve

# To make this permanent, add to your shell profile (~/.zshrc):
export OLLAMA_HOST=0.0.0.0:11434
```

Test from another Tailscale device:

```bash
# From your phone, laptop, or other device on the tailnet:
curl http://my-mac:11434/api/tags
# Expected: JSON list of your installed models

# Test inference
curl http://my-mac:11434/api/generate \
  -d '{"model": "qwen2.5:7b", "prompt": "Hello!", "stream": false}'
```

## Step 2: File sharing with Taildrop

Taildrop transfers files between Tailscale devices — like AirDrop but cross-platform and cross-network. Useful for sending large model files between machines without cloud storage.

```bash
# Send a file to another device on your tailnet
tailscale file cp large-model.gguf my-linux-server:

# Receive files sent to your Mac (they go to ~/Downloads by default)
tailscale file get ~/Downloads/

# Check pending transfers
tailscale file get --list
```

Taildrop transfers go peer-to-peer (direct WireGuard) when possible — no upload to a server. Transfer speeds are limited by your network connection between devices.

## Step 3: Subnet routing

Subnet routing lets other Tailscale devices access your entire local network (not just your Mac). This is useful for reaching a NAS, printer, or home server that doesn't have Tailscale installed.

```bash
# On your Mac: advertise your local subnet
sudo tailscale up --advertise-routes=192.168.1.0/24

# Approve in admin console: admin.tailscale.com → Machines → Edit route settings

# On another device: enable the route
tailscale up --accept-routes

# Now you can reach devices on your home LAN by their local IPs:
ping 192.168.1.100   # From your laptop, while away from home
ssh pi@192.168.1.150  # Access a Raspberry Pi on your home network
```

## Step 4: Access control configuration

By default, all devices on your tailnet can communicate with each other. For more control, configure ACLs at admin.tailscale.com → Access controls. This example restricts Ollama access to specific devices:

```json
{
  "acls": [
    {
      "action": "accept",
      "src": ["tag:client"],
      "dst": ["tag:ollama-server:11434"]
    }
  ],
  "tagOwners": {
    "tag:client": ["autogroup:member"],
    "tag:ollama-server": ["autogroup:admin"]
  }
}
```

Tag your Ollama Mac as `ollama-server` and your other devices as `client` in the admin console.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `tailscale status` shows "Stopped" | tailscaled daemon not running | `brew services restart tailscale` or open the Mac App |
| Devices can't reach each other despite both being online | ACL rules too restrictive | Check admin.tailscale.com → Access controls; default is allow-all |
| MagicDNS hostname not resolving | DNS override not active | Run `tailscale up --reset`; or try the `100.x.x.x` IP directly |
| SSH connection times out through Tailscale | macOS firewall blocking incoming connections | System Settings → Firewall → allow SSH (or check Remote Login is enabled) |
| High latency (100ms+) to nearby device | Using DERP relay instead of direct connection | Run `tailscale netcheck` to see relay quality; check if UDP is blocked |
| Tailscale breaks LAN internet access | Exit node accidentally enabled | `tailscale up --exit-node=` to disable the exit node |
| Login loop — browser keeps showing login page | Cookie/session issue | Try incognito window or different browser; or `tailscale logout && tailscale up` |

### Diagnosing connection quality

```bash
# Check what relay server you're using and latency to each
tailscale netcheck
# Shows: preferred DERP relay, latency to each relay, UDP accessibility

# Verify direct vs relay connection to a specific device
tailscale ping --c 5 other-device
# "via DERP" = relay; just latency = direct WireGuard

# Check Tailscale daemon logs for errors
log show --predicate 'subsystem == "com.tailscale.ipn.macos"' --last 5m
```

### Fixing "module not found" or daemon errors after macOS update

macOS updates sometimes require reinstalling network extensions:

```bash
# Remove and reinstall Tailscale
brew services stop tailscale
brew uninstall tailscale
brew install tailscale
brew services start tailscale
tailscale up
```

### Restricting Ollama to specific Tailscale devices

If you want only certain devices to access your Ollama server (not everyone on your tailnet), bind Ollama to your Tailscale IP specifically rather than `0.0.0.0`:

```bash
# Get your Tailscale IP
MY_TS_IP=$(tailscale ip -4)

# Bind Ollama to Tailscale interface only
OLLAMA_HOST=${MY_TS_IP}:11434 ollama serve

# Now only Tailscale devices can reach Ollama (your LAN cannot)
```
