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

macOS has built-in SSH and screen sharing, letting you access your Mac remotely on your local network without any third-party software.

## Prerequisites

- macOS (any recent version)
- Local network or Tailscale for remote access

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** Low — enable built-in macOS features

<!-- tab: SSH Setup -->
## Enable SSH

System Settings → General → Sharing → Remote Login → Enable

## Find your IP address

```bash
ipconfig getifaddr en0      # Wi-Fi
ipconfig getifaddr en1      # Ethernet
```

## Connect via SSH

From another machine:
```bash
ssh username@192.168.1.xxx
# Or use Bonjour (local network):
ssh username@your-mac.local
```

## SSH key setup (passwordless login)

```bash
# On the client machine:
ssh-keygen -t ed25519 -C "your-email@example.com"
ssh-copy-id username@your-mac.local
```

<!-- tab: Screen Sharing -->
## Enable Screen Sharing

System Settings → General → Sharing → Screen Sharing → Enable

## Connect via Screen Sharing

From another Mac: Finder → Go → Connect to Server → `vnc://your-mac.local`

Or from the Finder: Click on your Mac under Network in the sidebar.

## Command line remote management

Enable Remote Management instead of Screen Sharing for full control:
System Settings → General → Sharing → Remote Management → Enable

<!-- tab: Tips -->
## Access from outside your network

Use Tailscale (see the Tailscale playbook) for secure remote access from outside your local network.

## Wake on LAN

Enable in System Settings → Energy → Wake for network access.

## Check active SSH connections

```bash
who
# or
ss -tn sport = :22
```

## SSH config file for easy access

Add to `~/.ssh/config`:
```
Host my-mac
  HostName your-mac.local
  User your-username
  IdentityFile ~/.ssh/id_ed25519
```

Then connect with just: `ssh my-mac`
