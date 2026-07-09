# Runbook — NVIDIA Driver Setup for B200 (sm_100): Open Kernel Module Requirement

**Audience:** Lab engineers preparing a B200 host for the XHBM PoC (Ubuntu 22.04/24.04)
**Maintainer:** UmpaRumpa Inc. — PoC Item 8 (Adoption Path Validation) companion document
**Status:** v1.0
**Last verified:** 2026-07-07 (H100, driver 580.126.20 / CUDA 13.0, full timestamped transcript); 2026-05-27/28 (B200 SXM6, commits 1e283f9 / 50ec7e3 / b867ab2)

---

## TL;DR (3 lines)

1. B200 (Blackwell, `sm_100`) requires the **open kernel module** driver flavor — `nvidia-driver-580-server-open` (or a newer open variant).
2. If the **closed/proprietary DKMS driver is present on the same host**, it conflicts with the open module and the GPU will not initialize correctly for sm_100 workloads. The two flavors cannot coexist.
3. Fastest path: follow **§4 Prevention** on a clean host. If the host already has a closed driver, go straight to **§5 Remediation**.

## 1. Why this document exists

We hit this exact trap ourselves while bringing up our first B200 in May 2026, and it cost us real hours. Per our working principle of disclosing the unfavorable first: we did not keep a full terminal transcript of that May session — a lesson in itself, and the reason every session since (including the 2026-07-07 rehearsal on a bare GPU Base image) has been recorded end-to-end with timestamped `script` logging. This runbook distills the fix path so your lab does not lose the hours we lost, and we promised it in our 2026-07-02 answer mail.

This is part of the Item 8 deliverable set: once the driver layer is correct, everything else the PoC needs arrives inside our container — Docker + NVIDIA Container Toolkit are the only host-side prerequisites.

## 2. Scope

- **Applies to:** B200 / Blackwell-class GPUs (`sm_100`) on Ubuntu 22.04 LTS or 24.04 LTS. Our own validation was on Ubuntu 22.04.
- **Also consistent with:** H100 (`sm_90`) — the same open driver (580.126.20, reporting CUDA 13.0) passed our 2026-07-07 rehearsal on a stock GPU Base 22.04 image, so a fleet standardizing on the open flavor covers both generations.
- **Host CUDA toolkit is NOT required.** Our stack container carries its own CUDA userspace (base image `nvidia/cuda:12.4.0-base-ubuntu22.04`); the newer host driver is backward-compatible with it. The host needs the driver only.
- **Out of scope:** Windows, RHEL, driver series older than 580.

## 3. Root cause (one paragraph)

Blackwell (`sm_100`) is supported through NVIDIA's **open GPU kernel modules** driver flavor. The classic closed/proprietary DKMS driver builds kernel modules with the same names from a different source; when both are present on a host, the wrong flavor can end up loaded, and the GPU is then unusable for sm_100 even though the system superficially looks like it has an NVIDIA driver. The fix is therefore not "install the open driver" but "**ensure only the open driver exists on the host**."

## 4. Prevention — clean-host install path (recommended)

**Step 4.1 — Confirm no closed driver is present**

```
dpkg -l | grep -i nvidia
```

Expected on a clean host: no closed-flavor `nvidia-driver-*` / `nvidia-dkms-*` packages. If closed packages appear, stop here and use §5 instead.

**Step 4.2 — Install the open kernel module driver**

```
sudo apt-get update
sudo apt-get install -y nvidia-driver-580-server-open
sudo reboot
```

**Step 4.3 — Verify** (§6, all three checks)

**Step 4.4 — Container prerequisites (Item 8 baseline)**

The only host-side additions our stack needs:

```
sudo apt-get install -y docker.io nvidia-container-toolkit
sudo usermod -aG docker $USER
```

Note from our 2026-07-07 rehearsal: on the cloud "GPU Base 22.04" image, Docker 29.2.1 and nvidia-container-toolkit were already preinstalled and only the `usermod` line was needed — worth checking your images for the same, it saves ~10 minutes on Day 1. Log out and back in after `usermod` so group membership applies.

## 5. Remediation — host already has the closed driver

**Step 5.1 — Identify what is installed and what is loaded**

```
dpkg -l | grep -i nvidia
lsmod | grep nvidia
dkms status
```

**Step 5.2 — Remove the closed flavor completely** (a partial removal is the classic way this trap re-bites: leftover DKMS entries rebuild the closed module on the next kernel update)

```
sudo apt-get purge -y 'nvidia-driver-*' 'nvidia-dkms-*' 'libnvidia-*'
sudo apt-get autoremove -y
```

**Step 5.3 — Install the open flavor and reboot**

```
sudo apt-get update
sudo apt-get install -y nvidia-driver-580-server-open
sudo reboot
```

**Step 5.4 — Verify** (§6). If verification still shows the wrong flavor, re-run `dkms status` and remove any remaining closed-flavor entries, then reboot again.

## 6. Verification checklist (pass = all three)

**Check 1 — Driver present and version sane**

```
nvidia-smi
```

Expect: driver **580.126.20** or newer open variant; the reported CUDA version will be 13.0+ (this is the driver's supported ceiling, not a host toolkit).

**Check 2 — Open kernel module (not closed) is the one loaded**

```
modinfo nvidia | grep -i license
```

Expect: a **Dual MIT/GPL** license string — the open module's signature. The closed module reports a proprietary "NVIDIA" license. This is the reliable discriminator, since both flavors present the same module name.

**Check 3 — GPU visible from inside a container (the gate that matters for this PoC)**

```
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

Expect: the same driver reading from inside the container. This uses the same base image as our stack container (`Dockerfile.xhbm`), and it is the exact gate our 2026-07-07 Item 8 rehearsal passed — full timestamped transcript available (`session_0707.log`; container GPU test PASS at driver 580.126.20 / CUDA 13.0).

## 7. Verified environments (our measurements)

| Date | GPU | OS image | Driver / CUDA reported | Result |
|---|---|---|---|---|
| 2026-05-27/28 | 1× B200 SXM6 (sm_100) | Ubuntu 22.04 | 580-series open flavor | Full measurement sprint after resolving the closed-driver conflict (128K/256K memory characterization; commits 1e283f9, 50ec7e3, b867ab2) |
| 2026-07-07 | 1× H100 SXM5 | GPU Base 22.04 (stock, bare) | 580.126.20 / 13.0 | Item 8 rehearsal: container GPU test PASS, stack container build 15.6 s, rollback verified; end-to-end timestamped transcript |

Honest labeling: the May session predates our transcript-everything practice, so §5's remediation steps are the standard open-module migration path rather than a paste of our exact May commands; the July session is fully transcript-backed. During PoC Week 1 we will run this runbook as written on your P5336 host and attach the resulting transcript as the authoritative record.

## 8. If something still fails

Capture the outputs of the three §6 checks plus `dpkg -l | grep -i nvidia` and `dkms status`, and send them to us — we will turn any new failure mode into an addendum here. Divergence is a finding; we would rather document it than work around it silently.
