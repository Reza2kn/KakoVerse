# 💬 KAKOverse — Crisis Support Conversations (RL-Ready)

**TL;DR:** long, real crisis-support chats you can train on. 69 cities, decade vibes, rich personas, 3 style variants per turn → preference pairs → reward model → RL. No platitudes. No bouncing. Just staying. 🧠❤️

---

## 🤔 Why
Most “empathetic” datasets are short, simple, and same-y. We want **realistic**, **multi-turn** (10–20) conversations where a **Seeker** is in crisis and a **Supporter** actually helps — stays present, asks consent, and avoids unsafe advice.

---

## 🚀 What this does
- **Place & time aware:** 69 cities × decades (1920s→2020s) to shape tone and norms.  
- **Persona engine:** one Seeker persona per city with rich context (birth, formative years, values, traits, current life).  
- **Crisis focus:** stabilization first (grounding, validation, micro-choices). Suggestions only with consent.  
- **Training signal:** each Supporter turn has **3 variants** controlled by a `care_level` knob (cool → warm).  
- **Auto-judges → pairs:** tiny “judges” score empathy, presence, stabilization, and safety → **A/B pairs (~4k)**.  
- **RM → RL:** train a Reward Model on those pairs, then use it for RL to improve the Supporter.

---

## 🧩 Persona schema (what we include)

- **`birth_context`** 🍼: decade, city, country, local vibe/notes  
- **`formative_context[]`** 🧭: places/decades the persona spent formattive years  in
- **`natural_tendencies`** 🎯: likely skills/hobbies that fit the region (home_cooking_persian🥘,poetry_writing🖋️,calligraphy✒️,street_photography📷,tea_hosting🫖, etc.) 
- **`schwartz_basic_values`** 🧩: 10 values (self_direction🧭,stimulation⚡,hedonism🍷,achievement🏆,power👑,security🔒,conformity📏,tradition📜,benevolence🤝,universalism🌍) (0–100)  
- **`hexaco`** 🌈: 6 traits (honesty_humility🤲,emotionality💓,extraversion🎉,agreeableness🤗,conscientiousness✅,openness_to_experience🌈) (0–100)  
- **`csi`** 🗣️: conversation style (directness, disclosure, etc.)  
- **`current_socio_demographics`** 🏙️: age, household, work, finances, social network, current location  
- **`presenting_problem`** 🚨: summary, symptoms, duration, triggers, risk flags (no graphic details)  
- **`meta`** 🔎: traceability (which cities/decades fed this)

> We keep it **plausible** (e.g., tendencies match region/history) and **useful** for tone & safety.

---

## 🗣️ Conversations
- **Roles:** `Seeker` (in crisis) × `Supporter` (stays, stabilizes)  
- **Length:** up to **20 turns**  
- **Variants:** per Supporter turn, **3** outputs via `care_level ∈ {0.2, 0.5, 0.8}`  
- **Scoring:** small verifiers per turn:  
  - **Empathy** (reflection/validation/consent/brevity)  
  - **Presence** (support vs referral ratio)  
  - **Stabilization** (calmer now vs prior turn)  
  - **Safety** (no self-harm instructions, no medical/legal directives, no fake hotlines)  
- **Pairs:** each turn → 3 head-to-heads → ~**4,140** pairs across 69 convos × 20 turns

---

🔒 Safety (hard lines)

No instructions for self-harm/violence/substances

No medical/legal directives or diagnoses

No fake hotlines or promises

Offer resources as options; do not abandon the user if they refuse

🧠 Why this is different

Long, grounded, diverse: multi-turn, place-and-time flavored, persona-rich

Crisis-first: stabilization beats “cheer up”

Trainable signal: built-in preference pairs → RM → RL

Swap-in judges: bring your own verifiers or alignment rules

🗺️ Roadmap

More languages + culture-aware style adapters

Better stabilization detector (beyond lexical proxies)

Human-in-the-loop audits on a small, spicy subset

Public RM checkpoints + minimal RL recipe

🤝 Contribute

PRs welcome — add cities/decades, personas, verifiers, or scoring tweaks. Keep it kind, safe, and practical.

📄 License

Apache 2.0
