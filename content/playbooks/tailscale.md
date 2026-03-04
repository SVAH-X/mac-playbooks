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

Tailscale creates a secure mesh VPN between your devices, letting you access your Mac from anywhere without exposing ports to the internet.

## Prerequisites

- macOS 10.13+
- A Tailscale account (free tier available)

## Time & risk

- **Duration:** 5 minutes
- **Risk level:** None

<!-- tab: Install -->
## Option 1: Mac App (recommended)

Download from https://tailscale.com/download/mac or the Mac App Store.

## Option 2: Homebrew

```bash
brew install tailscale
```

<!-- tab: Connect -->
## Authenticate

```bash
tailscale up
```

This opens a browser window to authenticate with your Tailscale account.

## Check status

```bash
tailscale status
```

## Get your Tailscale IP

```bash
tailscale ip -4
```

<!-- tab: Usage -->
## Access your Mac remotely

Once connected, you can SSH using the Tailscale IP from any device on your tailnet:

```bash
ssh username@100.x.x.x
# or using MagicDNS hostname:
ssh username@your-mac.tail12345.ts.net
```

## Share your Ollama server

Expose Ollama to other devices on your tailnet:

```bash
# Start Ollama bound to Tailscale interface
OLLAMA_HOST=0.0.0.0:11434 ollama serve
```

Then from another device: `curl http://your-mac.tail12345.ts.net:11434/api/tags`

## Exit node (route internet through your Mac)

```bash
# On your Mac, advertise as exit node
tailscale up --advertise-exit-node

# On other devices, use your Mac as exit node
tailscale up --exit-node=your-mac
```
