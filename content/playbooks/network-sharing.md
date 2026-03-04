---
slug: network-sharing
title: "macOS Network Sharing"
time: "5 min"
color: green
desc: "SSH, screen sharing, and remote access setup"
tags: [tools, networking]
spark: "Set Up Local Network Access"
category: tools
featured: false
whatsNew: false
---

<!-- tab: Overview -->
## Basic idea

macOS includes a built-in SSH server (`sshd`), a VNC server (Screen Sharing), and Bonjour for local hostname resolution — all built into the OS and disabled by default. Enabling them through System Settings → Sharing takes under a minute. With proper SSH key authentication, you get secure, passwordless terminal access to your Mac from any computer on your local network.

For access from outside your home network (over the internet), pair this setup with Tailscale (see the Tailscale playbook) — Tailscale provides the secure tunnel without any router port-forwarding.

## What you'll accomplish

Passwordless SSH access to your Mac using an ed25519 key pair (no password prompts), Screen Sharing enabled for full desktop control, and an SSH config file so you can connect with a single word (`ssh my-mac`) instead of typing the full hostname and username.

## What to know before starting

- **SSH (Secure Shell)**: An encrypted protocol for terminal access to a remote computer. When you SSH into your Mac, you get a Terminal session running on the Mac, from which you can run any command. The connection is fully encrypted.
- **SSH key authentication**: Instead of typing a password, you prove your identity using a cryptographic keypair. The private key stays on your client machine (never shared). The public key is installed on the server (your Mac). The server encrypts a challenge with your public key — only someone with the private key can decrypt it. This is both more secure and more convenient than passwords.
- **mDNS/Bonjour**: Apple's zero-configuration local network service. It lets you use `your-mac.local` instead of memorizing an IP address. The `.local` domain is resolved by the mDNS daemon (`mDNSResponder`) on your local network — it doesn't use the internet.
- **VNC (Virtual Network Computing)**: The protocol behind Screen Sharing. It transmits the remote screen as a bitmap image and forwards keyboard/mouse input. Less efficient than SSH but gives you a full desktop GUI.
- **Port forwarding over SSH**: SSH can tunnel other network ports through the encrypted connection. For example, forward your remote Mac's Ollama port to your local machine: `ssh -L 11434:localhost:11434 my-mac` — then `localhost:11434` on your machine talks to Ollama on the Mac.

## Prerequisites

- macOS (any modern version — Ventura 13+ for the System Settings UI shown below)
- Admin access to System Settings
- A second computer or device for testing (or another Terminal window)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** Low — enables built-in macOS features that are off by default; easily reversed by toggling the same switches off

<!-- tab: SSH Setup -->
## Step 1: Enable Remote Login (SSH server)

"Remote Login" in System Settings is macOS's name for the SSH server daemon (`sshd`). Enabling it starts `sshd` listening on port 22 and configures the firewall to allow incoming connections. You can restrict access to specific user accounts for extra security.

```
System Settings → General → Sharing → Remote Login → toggle ON
```

Optionally restrict to specific users:
```
Click the "i" next to Remote Login → change "All users" to "Only these users" → add your account
```

Verify the SSH daemon is listening:

```bash
# On your Mac, confirm sshd is running
sudo lsof -i :22
# Expected: sshd listed with LISTEN state

# Or check with netstat
netstat -an | grep "\.22 "
```

## Step 2: Find your Mac's IP address and hostname

You can connect to your Mac using either its IP address or its `.local` hostname (via Bonjour). On most Macs: `en0` = Wi-Fi, `en1` = Ethernet (though this varies by Mac model). The `.local` hostname works anywhere on your local network — no need to look up the IP address.

```bash
# Wi-Fi IP address
ipconfig getifaddr en0
# Example: 192.168.1.42

# Ethernet IP address
ipconfig getifaddr en1

# Your Mac's Bonjour hostname (no IP lookup needed)
scutil --get LocalHostName
# Example: my-mac
# Connect as: my-mac.local
```

The IP address changes when your router reassigns it. The `.local` hostname is stable (until you rename the Mac). Prefer the `.local` hostname for ongoing use.

## Step 3: Test SSH with password authentication

From another computer on the same network, connect with your macOS username and password. The first connection shows a host fingerprint — verify it matches what you see on the Mac to confirm you're connecting to the right machine.

```bash
# From another machine on your network
ssh your-username@your-mac.local

# First connection shows: host fingerprint verification
# The authenticity of host 'your-mac.local' can't be established.
# ED25519 key fingerprint is SHA256:xxxxxxxxxxxx.
# Are you sure you want to continue connecting (yes/no)? yes

# After accepting: you should get a shell prompt on your Mac
your-mac:~ your-username$
```

Verify the fingerprint on your Mac:
```bash
ssh-keygen -l -f /etc/ssh/ssh_host_ed25519_key.pub
# Should match what you saw during first connection
```

Type `exit` to close the session and return to your client machine.

## Step 4: Set up SSH key authentication (passwordless login)

Key authentication eliminates passwords — faster to connect and immune to brute-force attacks. Generate the key on your client machine (the one you're connecting from), then install the public key on your Mac.

`ed25519` is the modern choice over `rsa`: smaller keys, faster operations, and considered more secure.

```bash
# === On your CLIENT machine (the one you connect FROM) ===

# Generate an ed25519 keypair (if you don't have one already)
# The comment (-C) is just a label — use your email or machine name
ssh-keygen -t ed25519 -C "your-email@example.com"
# Press Enter to accept default location (~/.ssh/id_ed25519)
# Enter a passphrase (recommended) or leave blank for truly passwordless

# Copy your public key to your Mac
ssh-copy-id your-username@your-mac.local
# Prompts for your Mac password one final time
# After this: no more password prompts

# Test passwordless login
ssh your-username@your-mac.local
# Should connect immediately without any password prompt
```

What `ssh-copy-id` does: appends your `~/.ssh/id_ed25519.pub` to `~/.ssh/authorized_keys` on the Mac, and sets the correct permissions (`600` on the file, `700` on `~/.ssh/`).

## Step 5: Harden SSH by disabling password authentication (optional)

Once key auth works, disable password authentication to prevent brute-force attempts. Only do this after confirming key auth works — if you lock yourself out, you'll need physical access to the Mac to fix it.

```bash
# === On your MAC ===

# Edit the SSH server configuration
sudo nano /etc/ssh/sshd_config

# Find and change (or add) these lines:
PasswordAuthentication no
ChallengeResponseAuthentication no

# Save (Ctrl+X, Y, Enter), then restart sshd
sudo launchctl unload /System/Library/LaunchDaemons/ssh.plist
sudo launchctl load /System/Library/LaunchDaemons/ssh.plist

# Test from your client: should still connect (via key)
ssh your-username@your-mac.local
```

<!-- tab: Screen Sharing -->
## Step 1: Enable Screen Sharing

Screen Sharing in macOS uses the VNC protocol. Enabling it starts a VNC server listening on port 5900. "Remote Management" is a superset of Screen Sharing that adds Apple Remote Desktop features (remote software installation, system info queries) — enable Remote Management if you need those features; Screen Sharing is sufficient for GUI access.

```
System Settings → General → Sharing → Screen Sharing → toggle ON
```

Note the computer address shown in the settings: `vnc://your-mac.local` — you'll use this to connect.

You can optionally set a VNC password for non-Apple clients:
```
Click the "i" next to Screen Sharing → "VNC viewers may control screen with password" → set password
```

## Step 2: Connect from another Mac

The Screen Sharing app is built into macOS. Finder's "Connect to Server" opens it automatically for VNC URLs.

```
Finder → Go menu → Connect to Server...
Type: vnc://your-mac.local
Click Connect
```

Alternatively, use the Screen Sharing app directly:
```bash
open /System/Library/CoreServices/Screen\ Sharing.app
# Or spotlight: Cmd+Space → "Screen Sharing"
```

You'll see your Mac's desktop. Keyboard and mouse input are forwarded. The session works on your local network — combine with Tailscale for remote access.

## Step 3: Connect from non-Mac clients

For Windows, Linux, iOS, or Android clients, you need a VNC client and the VNC password (set in Step 1):

| Platform | Recommended Client | Notes |
|---|---|---|
| iOS/iPadOS | Screens 5 (paid) or VNC Viewer (free) | VNC Viewer is free from RealVNC |
| Windows | TigerVNC (free) | Open-source, actively maintained |
| Linux | Remmina (built-in on Ubuntu) | Also supports RDP and SSH |
| Android | VNC Viewer | Same as iOS |

Connection details for all clients:
- **Server**: `192.168.1.42` (your Mac's IP) or hostname if client supports mDNS
- **Port**: `5900` (default VNC port)
- **Password**: the VNC password you set in Screen Sharing settings

## Step 4: Performance tuning for slow connections

VNC over Wi-Fi can be slow for high-resolution Retina displays. Reduce the quality to improve responsiveness:

```bash
# Check your Mac's display resolution (this is what VNC transmits)
system_profiler SPDisplaysDataType | grep Resolution
# Example: Resolution: 2560 x 1600 Retina

# To reduce VNC bandwidth: lower the display resolution while screen sharing
# System Settings → Displays → choose a lower resolution during VNC sessions
```

For built-in Apple Screen Sharing, quality adapts automatically. For third-party VNC clients, look for "Color depth" settings — 8-bit color is 8x more efficient than 24-bit with acceptable quality for GUI tasks.

<!-- tab: Tips -->
## SSH config file for one-word connections

The `~/.ssh/config` file defines SSH connection shortcuts. Instead of `ssh your-username@your-mac.local -i ~/.ssh/id_ed25519`, you type `ssh my-mac`.

```bash
# Create or append to ~/.ssh/config
cat >> ~/.ssh/config << 'EOF'

Host my-mac
    HostName your-mac.local
    User your-username
    IdentityFile ~/.ssh/id_ed25519
    ServerAliveInterval 60      # Send keepalive every 60s to prevent session drops
    ServerAliveCountMax 3       # Drop connection after 3 missed keepalives

# For Tailscale access (from outside your network)
Host my-mac-remote
    HostName your-mac.tail1234.ts.net
    User your-username
    IdentityFile ~/.ssh/id_ed25519
EOF

chmod 600 ~/.ssh/config

# Test the alias
ssh my-mac
```

The `ServerAliveInterval 60` setting prevents SSH sessions from dropping when idle — particularly useful for long-running commands or `tmux` sessions.

## Port forwarding: access remote services locally

SSH port forwarding tunnels a remote port to your local machine. This is the secure way to access Ollama on your Mac from a remote machine's browser without opening any firewall ports.

```bash
# Forward Mac's Ollama (port 11434) to your local machine
# After running this, localhost:11434 on your laptop = Ollama on your Mac
ssh -L 11434:localhost:11434 my-mac -N &

# Now from your laptop's browser or another program:
curl http://localhost:11434/api/tags
# This request goes to your Mac's Ollama

# Stop the tunnel
kill %1   # Or find the PID with: jobs
```

For persistent tunneling, add `-L 11434:localhost:11434` to the `Host my-mac` block in `~/.ssh/config`.

## tmux for persistent SSH sessions

Without tmux, your running processes stop when your SSH connection drops. With tmux, processes run inside a tmux session that persists even if the SSH connection drops. You reconnect and reattach.

```bash
# Install tmux
brew install tmux

# === On your Mac (via SSH) ===

# Start a named tmux session
tmux new-session -s ml-training

# Run your long-running task
python train.py --epochs 100

# Detach from session (session keeps running): Ctrl+B, then D

# === Your SSH drops or you disconnect ===

# === Reconnect via SSH ===
ssh my-mac

# Reattach to the running session
tmux attach -t ml-training
# Your training is still running with all output preserved
```

## SFTP for file transfer

SSH File Transfer Protocol uses the SSH connection for file transfer — no separate configuration needed.

```bash
# Interactive SFTP session
sftp my-mac
sftp> ls                          # List remote files
sftp> get model_weights.pth       # Download from Mac
sftp> put local_dataset.tar.gz    # Upload to Mac
sftp> exit

# Non-interactive: download a specific file
scp my-mac:~/models/qwen.gguf ./local-models/

# Non-interactive: upload a directory
scp -r ./training-data/ my-mac:~/datasets/
```

GUI clients that work over SFTP: **Cyberduck** (free, macOS/Windows), **Transmit** (paid, macOS) — both support drag-and-drop file transfer to/from your Mac via SSH.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `ssh: connect to host your-mac.local port 22: Connection refused` | Remote Login not enabled | System Settings → Sharing → Remote Login → ON |
| Password authentication keeps failing | Caps lock, trailing space, or wrong username | Verify username with `whoami` on the Mac; check Caps Lock |
| Key authentication fails after setup | Wrong permissions on `authorized_keys` | On Mac: `chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys` |
| `your-mac.local` not resolving | mDNS not working or different network | Use IP address directly; check both devices are on same Wi-Fi network |
| Screen Sharing shows "Connection failed" | macOS firewall blocking port 5900 | System Settings → Firewall → Options → allow Screen Sharing |
| SSH session drops after a few minutes of idle | No keepalive configured | Add `ServerAliveInterval 60` to `~/.ssh/config` |
| Two devices have the same `.local` hostname | Bonjour hostname collision | Rename one: System Settings → General → Sharing → Local hostname |

### Fixing SSH key permission errors

SSH is strict about file permissions — if your private key or `.ssh/` directory has permissions that are too open, SSH refuses to use them:

```bash
# Fix permissions on client machine
chmod 700 ~/.ssh
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/config

# Fix permissions on your Mac (server)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

### mDNS troubleshooting

If `your-mac.local` doesn't resolve, the mDNS service may be disrupted:

```bash
# Check if mDNSResponder is running on your Mac
pgrep -x mDNSResponder
# Should return a PID number

# Flush mDNS cache (restarts the daemon)
sudo dscacheutil -flushcache
sudo killall -HUP mDNSResponder

# Test mDNS resolution from another machine
dns-sd -q your-mac.local   # Should return 192.168.x.x
# Or simply:
ping your-mac.local
```

If mDNS consistently fails, use the IP address. Set a static IP on your Mac in System Settings → Wi-Fi → your network → Configure IP → Manually, then use that IP in your SSH config.

### Reconnecting after IP address change

If your Mac's IP changes (DHCP reassignment), `ssh your-mac.local` still works but `ssh 192.168.1.x` breaks. Always prefer the `.local` hostname for LAN access. For internet access where `.local` doesn't work, use Tailscale, which provides a stable `100.x.x.x` address that never changes.
