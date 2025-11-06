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
- **Auto-judges → pairs:** tiny “judges” score empathy, presence, stabilization, and safety → preference pairs for reward modelling.  
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
- **Pairs:** each turn → 3 head-to-heads → scales with conversation count (30 conversations already produce thousands of preference pairs)

## 🧨 Crisis category catalog
Age assignments now obey life-stage and persona-context boundaries. Each persona’s crisis profile (ages 20–100) is filled with summaries drawn from the following 45 categories*:

`Anger Management Issues`, `Anxiety Disorders`, `Bipolar Disorder`, `Death of a Loved One`, `Emotional Fluctuations`, `Grief and Loss`, `Identity Crises`, `Obsessive-Compulsive Disorder (OCD)`, `Ongoing Depression`, `Post-Traumatic Stress Disorder (PTSD)`, `Schizophrenia`, `Self-Esteem Issues`, `Spirituality and Faith`, `Sexual Orientation`, `Sexual Assault or Domestic Violence Recovery`, `Academic Pressure`, `Burnout`, `Chronic Stress`, `Financial Problems`, `Health Problems`, `Job Crisis`, `Life Transitions (e.g., Retirement, Relocation)`, `Workplace Stress`, `Breakups or Divorce`, `Conflicts or Communication Problems`, `Issues with Children`, `Issues with Parents`, `Marital Problems`, `Problems with Friends`, `School Bullying`, `Culture Shock`, `Appearance Anxiety`, `Career Development Issues`, `Goal Setting Issues`, `Motivation Problems`, `Personal Growth Challenges`, `Procrastination`, `Sleep Problems`, `Addictive Behaviors (e.g., Drug Use, Gambling)`, `Alcohol Abuse`, `Compulsive Behaviors`, `Eating Disorders`, `Internet Addiction`, `Self-Harm Behaviors`, `Debt Problems`.

*Category list inspired by the great work done by Shi et al., Findings 2025 [Beyond Coarse Labels: Fine-Grained Problem Augmentation and Multi-Dimensional Feedback for Emotional Support Conversation](https://aclanthology.org/2025.findings-emnlp.86/) 

Every category has an allowed age span (e.g., school bullying stays in the 20s, retirement transitions appear later in life), so crisis summaries remain coherent for each persona.

### ✅ Current progress
- **Life-stage crisis plan:** all personas share a 45-category grid with constrained age spans. Crisis summaries live in `artifacts/crisis_profiles/` with ages sorted 20→100.
- **Conversations in flight:** first **40 / 69** personas have full chats (see `artifacts/conversations/`). Generation script is resumable (`--offset` / `--limit` / `--skip-existing`) for future batches.
- **Supporter styles:** baseline, empathetic (positive signal), and cold (negative signal) responses captured per turn—ready for reward-model A/B training.

## 📦 Data & Repro
- **Examples:** small JSON samples live in `examples/` (`personas/` & `conversations/`) so docs/tests stay lightweight.
- **Full datasets:** generate locally or download published releases:
  - Personas → `Reza2kn/kakoverse-personas-v0` (Hugging Face Hub)
  - Crisis profiles → `Reza2kn/kakoverse-crisis-profiles-v0`
  - Conversations → `Reza2kn/kakoverse-conversations-v0`
- **Re-generate locally:** outputs land in `./artifacts` (gitignored).
  ```bash
  uv run python Scripts/generate_persona_plan.py
  uv run python Scripts/generate_crisis_category_plan.py
  uv run python Scripts/generate_crisis_profiles.py
  uv run python Scripts/generate_persona_cards.py
  uv run python Scripts/generate_conversations.py
  uv run python Scripts/analyze_conversations.py
  ```
- **Cleanup:** `rm -rf artifacts outputs logs` (or use the Makefile below).
- **Publish:** use `huggingface_hub.upload_folder` to push `artifacts/` contents to your dataset repos when ready.

### Publishing to the Hub

We keep generated corpora out of git; instead, publish them straight to the Hub:

```bash
uv run python Scripts/publish_to_hf.py \
  --personas-repo Reza2kn/kakoverse-personas-v0 \
  --crisis-repo Reza2kn/kakoverse-crisis-profiles-v0 \
  --conversations-repo Reza2kn/kakoverse-conversations-v0
```

The script reads `HF_TOKEN` from your environment (or `.env`) and automatically writes dataset cards. The first push bootstraps the repositories; subsequent runs upload only the changed examples thanks to the Hub’s chunked dataset storage.

### Quickstart for Exploration

- **Gradio browser:** `uv run python -m gradio_app.app` launches an interactive explorer. Click “Browse conversations” (defaults to `Reza2kn/kakoverse-conversations-v0`), then double-click any row to view the full dialogue with seeker/supporter message bubbles and care levels.
- **Programmatic browsing:** all datasets load with a single `load_dataset("Reza2kn/kakoverse-conversations-v0")`.
- **Generating more samples:** the Makefile targets mirror the production pipeline—run `make generate` then `make analyze`. Add `--provider hf --hf-model <model_id>` to generation scripts to route via Hugging Face Inference Endpoints (OpenRouter remains the default).

## 🛠️ Pipeline anatomy

1. **Persona plan** — `generate_persona_plan.py` samples cities/decades with a balancing plan.
2. **Crisis categories** — `generate_crisis_category_plan.py` assigns 45 fine-grained crisis types with age constraints.
3. **Profiles** — `generate_crisis_profiles.py` fills ages 20–100 with summaries aligned to the plan.
4. **Personas** — `generate_persona_cards.py` produces the full persona JSON with presenting problems.
5. **Conversations** — `generate_conversations.py` pairs Seeker/Supporter models (OpenRouter or HF Inference) and writes multi-turn dialogues, ensuring one unique age/category per persona.
6. **Analytics** — `analyze_conversations.py` outputs CSV/PNG summaries (stored under `artifacts/`).

All generation utilities accept `--provider openrouter|hf`, so you can switch between OpenRouter (default) and any Hugging Face Inference Endpoint that supports `chat_completion`.

### Hosting the Gradio browser on Spaces

1. Create a new Space (SDK = “Gradio”) under your account.
2. Drop `gradio_app/app.py` into the Space (or point to this repo).
3. Add a `requirements.txt` with `gradio` and `datasets` if the Space doesn’t use `uv` packaging.
4. The Space will present the same browse experience: load a dataset, double-click a row, inspect the chat bubbles with care levels.

> Tip: if you hit rate limits, run the generator in persona batches with `--offset` / `--limit` and lower sleep intervals to fully utilize the free tier.

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

*Coming soon:* persona-native language conversations (e.g., isiZulu for a Durban seeker, Español for MexicoCity) once the way is paved with English. The crisis category plan already keeps timelines balanced—next we’ll swap in multilingual prompts + locale-sensitive tone adapters.

🤝 Contribute

PRs welcome — add cities/decades, personas, verifiers, or scoring tweaks. Keep it kind, safe, and practical.

📄 License

Apache 2.0
