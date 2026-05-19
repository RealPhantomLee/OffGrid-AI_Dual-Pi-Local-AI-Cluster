# Hardware Guide

This document covers the bill of materials, hardware specifications relevant to AI inference, and practical guidance for storage, power, cooling, and networking for the OffGridAI dual-node Raspberry Pi 5 cluster.

---

## Bill of Materials

| Component | Recommended Model | Approx. Price (USD) | Notes |
|---|---|---|---|
| Raspberry Pi 5 8GB (×2) | Raspberry Pi 5 8GB Rev 1.0 | $80 each | 4GB variant will NOT run Mistral 24B — insufficient RAM even across two nodes |
| Hailo-8 AI HAT | Raspberry Pi AI HAT+ (26 TOPS) | $70 | RPi 5 only — requires PCIe HAT+ connector not present on earlier Pi models |
| microSD 32GB+ Class A2 (×2) | SanDisk Extreme Pro A2 32GB | $10 each | NVMe SSD strongly recommended instead — see storage section |
| NVMe SSD 250GB (for aipi) | Samsung 980 250GB or WD Blue SN570 250GB | $25–40 | For aipi only if not using Hailo-8 (PCIe conflict — see below) |
| RPi 5 M.2 HAT (for NVMe) | Raspberry Pi M.2 HAT+ | $20 | Shares PCIe lane with Hailo-8 — cannot use both on the same Pi simultaneously |
| Official RPi 5 27W USB-C PSU (×2) | Raspberry Pi 27W USB-C Power Supply | $12 each | Mandatory — third-party PSUs that cannot sustain 5A cause random crashes and filesystem corruption |
| Active cooler (×2) | Argon ARTIC ($25) or Official RPi 5 Active Cooler ($5) | $15–25 each | Required for sustained inference — passive cooling is insufficient |
| Gigabit network switch | TP-Link TL-SG108 (8-port unmanaged) | $20 | Wired connection is mandatory for RPC inference — WiFi causes inference stalls |
| Ethernet cables (×2) | Cat5e or Cat6, 1m | $5 each | Flat cables are convenient for tight enclosures |
| RPi 5 cases (×2) | Argon NEO 5, official RPi 5 case, or equivalent | $10–15 each | Ensure case is compatible with active cooler and any HAT |

**Total estimated cost: $350–450** depending on case selection and whether NVMe or SD card is used.

Optional additions:
- UPS (CyberPower CP425SLG or APC BE425M): $45–60. Strongly recommended to prevent SD card and NVMe filesystem corruption on unexpected power loss.
- USB microphone + speaker: for voice pipeline integration with Whisper STT.
- Additional Pi 5 8GB (third node): $80. Enables 34B models with 24 GB combined RAM.

---

## Raspberry Pi 5 8GB — Inference-Relevant Specifications

| Spec | Value | Relevance to AI Inference |
|---|---|---|
| CPU | 4× ARM Cortex-A76 @ 2.4 GHz | Generates approximately 2–6 tokens/sec on Mistral 24B Q4_K_M |
| RAM | 8 GB LPDDR4X-4267 | Memory bandwidth is the primary bottleneck for LLM inference on CPU |
| PCIe interface | Gen 2.0 x1 via HAT+ connector | Required for Hailo-8 accelerator and NVMe M.2 HAT — only one device at a time |
| ISA extensions | NEON, ARM FMA, DOTPROD | llama.cpp uses these SIMD extensions for quantised matrix multiply, yielding 40%+ speedup vs scalar |
| Power input | 5V / 5A via USB-C | Requires official 27W PSU under full LLM load — insufficient supply voltage causes thermal throttling and crashes |
| Thermal throttle threshold | 80°C (emergency shutdown at 85°C) | Active cooling is required for sustained inference — CPU reaches 75–82°C without a fan |
| Storage interfaces | microSD, USB 3.0, PCIe 2.0 x1 (M.2) | microSD is sufficient for boot; NVMe dramatically improves model load time and longevity |
| Ethernet | Gigabit (1000BASE-T) | Required for llama.cpp RPC inference between nodes |
| USB | 2× USB 3.0, 2× USB 2.0 | USB 3.0 external SSD is a viable storage alternative if PCIe is occupied by Hailo-8 |

The Cortex-A76 cores perform significantly better than the A72 cores in Raspberry Pi 4. The DOTPROD extension (dot product of int8 vectors) is directly exploited by llama.cpp's ARM-specific quantised matrix multiplication routines, and is a primary reason why the Pi 5 achieves 2–6 t/s rather than the ~0.8–1.5 t/s observed on the Pi 4 for comparable models.

---

## Hailo-8 AI HAT

### Physical Installation

The Hailo-8 AI HAT connects to the Raspberry Pi 5 via the HAT+ connector on the board's top edge. It uses the PCIe 2.0 x1 interface exposed by the RP1 I/O controller. The Hailo-8 chip itself sits on a custom PCB in an M.2 2242 form factor mounted to the HAT carrier board. A short FPC (Flexible Printed Circuit) ribbon cable connects the M.2 slot on the HAT to the Pi's HAT+ connector. The HAT includes a heatsink on the Hailo-8 chip.

Installation steps:
1. Attach the FPC ribbon cable to the HAT+ connector on the Pi 5 (zero-insertion-force connector, latch upward to insert).
2. Mount the Hailo-8 HAT above the Pi using the provided standoffs.
3. Ensure the HAT's heatsink has adequate airflow — the Hailo-8 generates up to 4W during active inference.
4. Install the `hailort` driver package and enable the PCIe interface in `/boot/firmware/config.txt` (`dtparam=pciex1`).

### PCIe Bus Sharing — Critical Constraint

The Raspberry Pi 5 exposes **one** PCIe 2.0 x1 interface via the HAT+ connector. Both the Hailo-8 AI HAT and the Raspberry Pi M.2 HAT+ (for NVMe) use this same interface. **They cannot both be connected to the same Pi simultaneously** without a PCIe switch.

This creates a forced choice for the aipi node:

| Option | Storage | Vision | Trade-off |
|---|---|---|---|
| Hailo-8 HAT + SD card | microSD (slow, wears out) | 26 TOPS hardware vision | Fastest vision inference; shortest storage lifespan |
| M.2 HAT + NVMe | NVMe (fast, durable) | Software vision only via llava-phi3 | Best storage performance; vision is slower |
| PCIe switch HAT | NVMe | 26 TOPS hardware vision | Both capabilities; adds $30–50, increases complexity |

The current verified configuration uses Hailo-8 + microSD on aipi. jolly has no PCIe devices and can use the M.2 HAT with NVMe without any conflict. Consider putting NVMe on jolly to extend jolly's storage life, even though jolly's model storage role means it also benefits from fast NVMe access.

### Why the Hailo-8 Cannot Run LLMs

The Hailo-8 is a fixed-function dataflow processor optimised for convolutional neural networks and other feedforward architectures. It uses a spatial dataflow execution model where computation is mapped at compile time onto a mesh of processing elements. Models must be compiled to a proprietary `.hef` (Hailo Executable Format) file using the Hailo Dataflow Compiler before they can run on the device.

Transformer attention layers — the core of every LLM — require dynamic, data-dependent memory access patterns (the attention matrix varies with each input token). The Hailo-8's architecture cannot express this dynamism. There is no software path to run llama.cpp, Ollama, or any GGUF model on the Hailo-8; it is physically incapable of executing the attention mechanism.

### What the Hailo-8 Can Do

| Task | Model | Performance | Power |
|---|---|---|---|
| Object detection | YOLOv8n / YOLOv8s | 26 TOPS, real-time HD | ~2–4 W |
| Face detection | LightFace / RetinaFace | Real-time, high accuracy | ~2 W |
| Image classification | ResNet-50, MobileNetV2 | Real-time | ~1.5 W |
| Pose estimation | MoveNet, OpenPose | Real-time | ~3 W |
| Semantic segmentation | DeepLabV3 | Real-time | ~3.5 W |

These capabilities feed into the OffGridAI pipeline as structured JSON output injected into the LLM context, giving the language model awareness of what the camera sees without the LLM itself needing to process images.

---

## Storage Recommendations

### SD Card Limitations Under AI Workload

A Raspberry Pi 5 running the OffGridAI stack imposes two distinct write-amplification stressors on the SD card:

1. **Model loading:** The Mistral 24B Q4_K_M GGUF file is 14.3 GB. Every time llama-server restarts, the OS reads this entire file into RAM. On a Class A2 microSD this takes approximately 2–3 minutes and subjects the card to 14+ GB of sustained sequential reads on every restart.

2. **ChromaDB writes:** The vector database performs continuous small writes as documents are ingested and embeddings are updated. This creates a write-amplification pattern that is particularly damaging to NAND flash memory.

A typical consumer microSD card rated for 10,000 programme/erase cycles will wear out in 6–12 months of daily use under this combined workload. Symptoms of wear include filesystem corruption on power loss, increasing read errors, and eventual boot failure.

### NVMe SSD (Strongly Recommended)

Installing an NVMe SSD via the Raspberry Pi M.2 HAT+ reduces model load time from approximately 3 minutes (microSD) to approximately 30 seconds. It also dramatically extends storage lifespan: enterprise-grade NVMe SSDs are rated for 150+ TBW (terabytes written); a Pi AI workload writes perhaps 5–20 GB per day, giving years of endurance.

Recommended models:
- **Samsung 980 250GB** — well-tested on RPi 5, consistent performance, ~$30
- **WD Blue SN570 250GB** — reliable, good sequential performance, ~$28
- **Kingston NV3 250GB** — budget option, adequate for this workload, ~$25

Avoid QLC-based SSDs (e.g. some Samsung 870 QVO variants) for the ChromaDB write pattern — TLC and MLC NAND handle small random writes much better.

### PCIe Conflict and Options

If aipi uses the Hailo-8 AI HAT, the PCIe lane is occupied and the M.2 HAT cannot be installed. Options:

1. **SD card + Hailo-8 on aipi, NVMe + M.2 HAT on jolly.** jolly's model storage also benefits from NVMe. jolly has no PCIe devices competing for the interface.
2. **NVMe M.2 HAT on aipi, no Hailo-8.** Vision tasks handled by llava-phi3 running on CPU via Ollama. Slower vision inference but best storage configuration.
3. **PCIe switch HAT (e.g. Pimoroni PicoVision HAT or similar).** Adds a PCIe 2.0 switch to provide two downstream x1 ports from the Pi's single upstream port. Enables both Hailo-8 and NVMe simultaneously. Adds cost (~$30–50) and complexity; verify driver compatibility before purchasing.
4. **USB 3.0 NVMe enclosure on aipi.** Bypasses the PCIe conflict entirely by using USB 3.0 for the SSD. Sequential performance is slightly lower than native PCIe (USB 3.0 = ~400 MB/s vs PCIe M.2 ~1500 MB/s for this class of SSD) but still far superior to microSD, and the model load improvement is significant.

---

## Power Requirements

| Component | Power Draw |
|---|---|
| Raspberry Pi 5 at idle | 3–5 W |
| Raspberry Pi 5 under sustained LLM inference (100% CPU) | 8–12 W |
| Hailo-8 during active vision inference | 2–4 W additional |
| Both nodes under full load (worst case) | 20–28 W total |

### Why the Official PSU is Mandatory

The Raspberry Pi 5 requires a USB-C Power Delivery source capable of supplying **5V at 5A (25W)**. Under sustained 100% CPU inference load, the board draws up to 12W. The Hailo-8 adds another 2–4W. Third-party USB-C chargers and hubs frequently cannot sustain 5A without voltage droop — the Pi 5 will detect under-voltage (below 4.75V) and log a warning, then begin CPU throttling. In severe cases, the filesystem on the SD card can be corrupted if the voltage drops during a write operation.

The official Raspberry Pi 27W USB-C Power Supply is rated for exactly this use case. It is not overpriced convenience packaging — it has been characterised to maintain output voltage under the specific inrush and sustained load profile of the Pi 5. Use it.

### UPS Recommendation

A small uninterruptible power supply on aipi provides two benefits:
1. **Graceful shutdown on power loss:** The Pi 5 does not implement safe write-back on power loss. Any open database transaction in ChromaDB, any in-progress SD write, will be lost and potentially corrupt the filesystem. A UPS gives 5–15 minutes to run `shutdown -h now`.
2. **Voltage regulation:** Cheap UPS units act as line conditioners, protecting against brief voltage sags that could cause under-voltage events.

Recommended UPS models:
- **CyberPower CP425SLG** — 425VA / 255W, ~$45. At 25W draw, provides approximately 10–12 minutes of runtime for a graceful shutdown.
- **APC BE425M** — 425VA / 255W, ~$60. Similar capacity; APC has excellent Linux USB monitoring support via `apcupsd`.

Configure `apcupsd` or `NUT` (Network UPS Tools) to trigger automatic shutdown when battery reaches 50% — this gives 5+ minutes of shutdown time even on a deeply discharged battery.

---

## Thermal Management

### Why Active Cooling is Required

The Raspberry Pi 5 throttles CPU frequency when the SoC temperature exceeds 80°C and initiates emergency shutdown at 85°C. Under continuous LLM inference at 100% CPU utilisation, the Pi 5 in an open-air environment without active cooling reaches 75–82°C within 3–5 minutes. Once throttling begins, inference throughput drops significantly — a Pi that was generating 5 t/s can drop to 2 t/s under sustained thermal throttling.

The thermal throttle mechanism is a firmware-level hard limit. There is no software override. Active cooling is not optional for production inference workloads.

### Tested and Verified Coolers

| Cooler | Price | Notes |
|---|---|---|
| Official Raspberry Pi 5 Active Cooler | ~$5 | Official product, clips directly to SoC, controlled by firmware. Best value. |
| Argon ARTIC | ~$25 | Includes enclosure + fan. Quiet. Good airflow. Verified working with Hailo-8 HAT. |
| Pimoroni Fan SHIM | ~$10 | Low-profile fan. Requires GPIO connection. Works but fan control requires software setup. |
| Noctua NF-A4x10 (with adapter) | ~$15 | Quietest option. Requires custom mount or 3D-printed bracket. |

Any cooler with a fan is acceptable. Passive heatsinks alone are insufficient for sustained inference.

### Physical Placement Considerations

- Leave 2–3 cm clearance above the active cooler fan for exhaust air. Stacking two Pis directly on top of each other without a spacer will cause the lower Pi's exhaust to feed into the upper Pi's intake.
- The Hailo-8 HAT has a built-in heatsink but no fan. It requires airflow across its heatsink fins. Position aipi so the Hailo-8's heatsink is not in a stagnant air pocket.
- If rack-mounting both Pis in an enclosure, consider placing them side-by-side rather than vertically stacked, and ensure the enclosure has ventilation slots.

---

## Network Recommendations

### Why Wired Ethernet is Required for the RPC Node

The llama.cpp RPC protocol is latency-sensitive by design. During a transformer forward pass, activations must cross the network at each layer boundary between nodes. For Mistral 24B with 40 transformer layers distributed across two nodes, approximately 8–12 layer boundary crossings occur per generated token. Each crossing adds a round-trip network latency:

| Network type | Typical latency | Round-trip per token (est.) | Effect |
|---|---|---|---|
| Wired Gigabit Ethernet | ~0.5–1 ms | ~5–12 ms | Acceptable |
| WiFi 5 (802.11ac) | 2–10 ms typical, 50ms+ jitter | ~20–120 ms+ per token | Causes stalls, RPC timeouts |
| WiFi 6 (802.11ax) | 1–5 ms typical | ~10–60 ms per token | Still causes intermittent failures |

WiFi jitter is the critical failure mode — it is not average latency that breaks RPC inference, but the occasional 50–200 ms spike that causes the RPC client to time out and fail the entire inference pass. **Do not use WiFi for the jolly RPC node.** Run a physical Ethernet cable.

### Bandwidth Requirements

At 2–6 t/s generation speed with Mistral 24B Q4_K_M, the inter-node RPC bandwidth is approximately 50–150 MB/s during active inference (each activation tensor crossing is ~1–4 MB at 24B model size). Gigabit Ethernet (theoretical 125 MB/s) is sufficient. 2.5 Gigabit Ethernet approximately halves the network contribution to time-to-first-token and is worthwhile if a 2.5GbE switch is already available.

### Recommended Switch

**TP-Link TL-SG108** (8-port unmanaged Gigabit): $20. Unmanaged switches have no STP reconvergence delays, no management overhead, and no firmware update requirements. They simply forward frames. This is ideal for the RPC use case.

Avoid managed switches in the inference path unless you are confident that STP is disabled on the relevant ports. STP reconvergence events (30–50 seconds by default) will cause llama-server to fail its RPC connection and require a restart.

---

## Verified Working Configuration

The following is the exact hardware this project is developed and tested on:

**aipi (coordinator node):**
- Raspberry Pi 5 8GB Rev 1.0
- Hailo-8 AI HAT (26 TOPS, PCIe 2.0 x1)
- 32GB SanDisk Extreme A2 microSD (boot and model storage)
- Official Raspberry Pi 27W USB-C Power Supply
- Argon ARTIC active cooler + enclosure
- Raspberry Pi OS Bookworm (64-bit) / Arch Linux ARM

**jolly (worker node):**
- Raspberry Pi 5 8GB
- 32GB microSD (Class A2)
- Official Raspberry Pi 27W USB-C Power Supply
- Active cooler (official RPi 5 Active Cooler)
- Arch Linux ARM

**Network:**
- Both nodes connected via Gigabit Ethernet
- TP-Link 8-port Gigabit switch
- Cat6 patch cables

**Model in use:**
- `mistral-small-3.1-24b-instruct-2503-q4_k_m.gguf` (14.3 GB)
- Distributed across aipi and jolly via llama.cpp RPC
- Context window: 4096 tokens
- Generation speed: 2.5–5 t/s observed

This configuration has been validated for stable multi-hour inference sessions. The primary known limitation is microSD wear on aipi, which will require either NVMe (sacrificing Hailo-8) or a PCIe switch HAT to address long-term.
