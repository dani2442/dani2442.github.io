---
title: "Space Piracy: Theft, Ransom, and Security in Orbit"
date: 2025-12-23
tags: ["space", "satellites", "security", "cybersecurity", "economics"]
categories: ["space", "miscellaneous"]
author: "Daniel López Montero"
showToc: true
draft: true
description: "Why piracy is likely to move to orbit, and how to defend against it."
ShowWordCount: false
ShowReadingTime: true
comments: true
UseHugoToc: true
editPost:
    URL: "https://github.com/dani2442/dani2442.github.io/tree/main/content"
    Text: "Suggest Changes" # edit text
    appendFilePath: true # to append file path to Edit link
---

If the space industry continues to develop as expected, it's only a matter of time before space piracy becomes a real concept. All these expensive satellites and data centers orbiting look very juicy and largely unprotected!

![Boats](ship.png)

![Space Piracy](space_piracy.png)

But "piracy" is a loaded word. If you imagine a grappling hook and a boarding party, you're thinking about the hardest (and least likely) version. The first wave of "piracy" will look far more mundane: credential theft, supply-chain compromises, command spoofing, and subtle manipulation of telemetry. In other words, it will resemble today's cybercrime - just with higher stakes and stranger physics.

## What counts as space piracy?

I'll use *space piracy* as a broad umbrella: extracting value from a space asset without authorization by taking control of it, degrading it, or coercing its operator.

That includes several distinct behaviors:

- **Service theft**: using someone else's satellite capacity (bandwidth, compute, sensing) without paying.
- **Data theft**: exfiltrating high-value data products (imagery, signals intelligence, proprietary processing outputs) by compromising the ground segment or the downlink pipeline.
- **Ransom/denial**: threatening to disable, deorbit, or simply make an asset useless unless paid.
- **Signal hijacking**: interfering with user links, spoofing signals, or impersonating a satellite to users on the ground.
- **Physical capture or tampering**: rendezvous and proximity operations (RPO), docking, attaching a device, or relocating the satellite.

Most real-world incidents will be messy combinations: a cyber intrusion that enables a malicious maneuver; a jammed link that forces the operator into a risky recovery mode; a "salvage" attempt that is indistinguishable from theft.
