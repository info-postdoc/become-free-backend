"""
Become Free — RAG Backend
FastAPI server with TF-IDF retrieval + Dr. Clarisse's voice response engine
"""
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import json, re, os, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = FastAPI(title="Become Free RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load knowledge base ──────────────────────────────────────────────────────
KB_PATH    = "/home/sandbox/become_free/knowledge_base.json"
INDEX_PATH = "/home/sandbox/become_free/tfidf_index.pkl"

with open(KB_PATH, 'r') as f:
    chunks = json.load(f)

texts = [c['text'] for c in chunks]

# Build or load TF-IDF index
if os.path.exists(INDEX_PATH):
    with open(INDEX_PATH, 'rb') as f:
        vectorizer, tfidf_matrix = pickle.load(f)
    print("✅ TF-IDF index loaded from cache")
else:
    vectorizer  = TfidfVectorizer(ngram_range=(1,2), max_features=20000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    with open(INDEX_PATH, 'wb') as f:
        pickle.dump((vectorizer, tfidf_matrix), f)
    print(f"✅ TF-IDF index built: {tfidf_matrix.shape}")

# ── Safety guardrails ────────────────────────────────────────────────────────
BLOCKED_TOPICS = [
    r'\bdiagnos\w*\b', r'\bprescri\w*\b', r'\bmedication\b', r'\bdrug\b',
    r'\bsuicid\w*\b', r'\bself.harm\b', r'\bkill\w*\b', r'\bhurt\w*\b',
    r'\btherapist\b', r'\bpsychiatri\w*\b', r'\bcancel\w*\b',
    r'\bpolitics\b', r'\bpresident\b', r'\belection\b', r'\bsport\w*\b',
    r'\bstock\w*\b', r'\binvest\w*\b', r'\bcrypto\b', r'\bweather\b',
    r'\brecipe\b', r'\bcook\b',
]

CRISIS_WORDS = [r'\bsuicid\w*\b', r'\bself.harm\b', r'\bkill myself\b', r'\bend my life\b']

PROGRAM_TOPICS = [
    'nervous system', 'cortisol', 'inner child', 'emotional eating',
    'weight', 'body', 'trauma', 'healing', 'safe body', 'program',
    'phase', 'week', 'regulation', 'mindful', 'movement', 'sleep',
    'boundary', 'identity', 'freedom', 'dr clarisse', 'clarisse',
    'worksheet', 'exercise', 'tool', 'breathing', 'grounding',
    'somatic', 'polyvagal', 'leptin', 'ghrelin', 'hpa', 'shame',
    'food', 'eating', 'hunger', 'stress', 'anxiety', 'enroll',
    'apply', 'cohort', 'cost', 'price', 'session', 'community',
    'become free', 'survival', 'safety', 'nourishment', 'faith',
]

def is_crisis(text):
    t = text.lower()
    return any(re.search(p, t) for p in CRISIS_WORDS)

def is_blocked(text):
    t = text.lower()
    return any(re.search(p, t) for p in BLOCKED_TOPICS)

def is_on_topic(text):
    t = text.lower()
    return any(kw in t for kw in PROGRAM_TOPICS)

# ── Retrieval ────────────────────────────────────────────────────────────────
def retrieve(query, top_k=5):
    q_vec  = vectorizer.transform([query])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()
    top_idx = np.argsort(scores)[::-1][:top_k]
    results = []
    for idx in top_idx:
        if scores[idx] > 0.05:
            results.append({
                "text":  chunks[idx]['text'],
                "topic": chunks[idx]['topic'],
                "score": float(scores[idx])
            })
    return results

# ── Response Engine (Dr. Clarisse's voice) ──────────────────────────────────
def build_response(query, retrieved):
    q = query.lower().strip()

    # ── Greeting
    if any(w in q for w in ['hello','hi','hey','good morning','good evening','good afternoon','howdy']):
        return (
            "Hello, and welcome to *Become Free* 🌿\n\n"
            "I'm here to walk alongside you on your healing journey. Whether you have questions about "
            "The Safe Body program, need guidance on a regulation tool, or just want to understand "
            "why your body responds the way it does — I'm here for you.\n\n"
            "What's on your heart today?"
        )

    # ── Who is Dr. Clarisse
    if any(w in q for w in ['who is dr', 'who is clarisse', 'about dr', 'tell me about you', 'who are you','who built','who created','founder']):
        return (
            "Dr. Clarisse Djeumo Djukom, MD is the physician, healer, and founder of The Safe Body program. 🌿\n\n"
            "She is a Medical Doctor with specialized training in trauma-informed coaching, somatic experiencing, "
            "inner child therapy, nervous system regulation, and nutritional medicine. She is also a **Critical Care "
            "& Burn Fellowship graduate from the University of Texas Medical Branch**.\n\n"
            "But beyond her credentials — Dr. Clarisse *lived* this journey. She knows what it feels like to do "
            "everything right and still feel trapped in a body that won't cooperate. That personal experience is "
            "what makes The Safe Body unlike any other program.\n\n"
            "Her message to every woman: *\"The weight was never the problem. The weight was the symptom.\"*"
        )

    # ── What is the program
    if any(w in q for w in ['what is the safe body','what is this program','tell me about the program','what is become free','what do you offer']):
        return (
            "The Safe Body is a **12-week, physician-founded, trauma-informed group coaching program** "
            "designed for women who have tried every diet — and are ready to heal the *root cause* instead. 🌿\n\n"
            "The program is built on 3 phases:\n\n"
            "🟢 **Phase 1 — Safety** (Weeks 1–4): Regulate your nervous system, understand your stress biology, "
            "and begin to feel safe in your body for the first time.\n\n"
            "🟡 **Phase 2 — Healing** (Weeks 5–8): Meet and heal your inner child, transform your relationship "
            "with food, discover joyful movement, and restore your sleep.\n\n"
            "🔵 **Phase 3 — Freedom** (Weeks 9–12): Reclaim your identity, build your support system, and step "
            "into sustainable, embodied freedom.\n\n"
            "It is not a diet. It is not a meal plan. It is a *healing journey*.\n\n"
            "Would you like to learn more about a specific phase, or are you ready to apply?"
        )

    # ── How to enroll
    if any(w in q for w in ['enroll','apply','join','sign up','how do i','cost','price','how much','cohort','next cohort','start']):
        return (
            "We would love to welcome you into Cohort 2026! 🌿\n\n"
            "Here's how it works:\n\n"
            "1️⃣ **Apply** — Visit our Apply Now page and fill out a short application. There is no commitment required to apply.\n"
            "2️⃣ **Discovery Call** — Dr. Clarisse reviews every application personally and invites selected women to a private discovery call.\n"
            "3️⃣ **Enrollment** — If The Safe Body is the right fit, you'll receive your enrollment details and welcome packet.\n\n"
            "📌 **Cohorts are limited to 12 women** to ensure every participant receives the depth of personal attention this healing work requires.\n\n"
            "👉 You can apply at: **thesafebody.com/contact** or click *Apply Now* in the navigation above.\n\n"
            "Do you have any questions before you apply?"
        )

    # ── Nervous system / regulation
    if any(w in q for w in ['nervous system','regulate','regulation','polyvagal','green zone','red zone','grey zone','calm down','overwhelm','anxious','anxiety','triggered','trigger']):
        context = retrieved[0]['text'] if retrieved else ""
        return (
            "Your nervous system is not broken — it is doing exactly what it was designed to do. 🌿\n\n"
            "The Safe Body uses **Polyvagal Theory** to help you understand your three nervous system states:\n\n"
            "🟢 **Green Zone (Safe & Social)** — You feel calm, connected, and able to make thoughtful choices. "
            "This is where healing and sustainable weight loss happen.\n\n"
            "🔴 **Red Zone (Fight or Flight)** — Cortisol is elevated. Your body is in survival mode. "
            "Emotional eating, cravings, and reactivity are common here.\n\n"
            "⚫ **Grey Zone (Shutdown/Freeze)** — You feel numb, disconnected, or exhausted. "
            "Binge eating and emotional numbness often live here.\n\n"
            "**A quick regulation tool you can try right now:**\n"
            "Take a slow breath in for 4 counts... hold for 2... and exhale slowly for 6 counts. "
            "The extended exhale activates your parasympathetic nervous system — your body's natural calm response.\n\n"
            "In Week 4 of The Safe Body, you'll build your own personal *Regulation Menu* — a personalized "
            "toolkit of tools that work specifically for your nervous system.\n\n"
            "Would you like to learn more about nervous system regulation or another part of the program?"
        )

    # ── Inner child
    if any(w in q for w in ['inner child','little girl','childhood','young self','past','reparent','childhood trauma','childhood experience']):
        return (
            "This is one of the most tender — and most transformative — parts of The Safe Body. 🌿\n\n"
            "Many women who struggle with emotional eating are not struggling with *food*. "
            "They are struggling with a little girl inside them who learned, very early, that food meant "
            "safety, comfort, or love — because no other source of safety was available.\n\n"
            "That little girl was not weak. She was *surviving*.\n\n"
            "In **Week 3 of The Safe Body**, Dr. Clarisse guides you through:\n"
            "✦ Meeting your inner child with compassion (not judgment)\n"
            "✦ Understanding how her survival strategies became your adult patterns\n"
            "✦ Writing a letter to her — one of the most powerful healing tools in the program\n"
            "✦ Beginning the process of reparenting — giving her the safety she never had\n\n"
            "*\"When I healed the little girl who learned to use food to feel safe, the woman in the mirror "
            "finally started to look like someone I recognized — and loved.\"* — Dr. Clarisse\n\n"
            "Would you like to explore this further, or learn about another part of the healing journey?"
        )

    # ── Emotional eating
    if any(w in q for w in ['emotional eating','eat when stressed','stress eating','binge','binge eating','food and emotions','eat my feelings','comfort food','crave','craving']):
        return (
            "Emotional eating is not a character flaw. It is a *nervous system response*. 🌿\n\n"
            "When your body is in survival mode — flooded with cortisol — your prefrontal cortex "
            "(the part that makes rational decisions) goes offline. Your brain's reward system takes over, "
            "and it reaches for the fastest source of dopamine available: food.\n\n"
            "You were never *weak*. Your biology was doing exactly what it was programmed to do under stress.\n\n"
            "In **Week 5 of The Safe Body**, you'll learn:\n"
            "✦ How to identify your specific emotional eating triggers\n"
            "✦ The difference between physical hunger and emotional hunger\n"
            "✦ How to pause between the trigger and the behavior — creating space for a new choice\n"
            "✦ How to meet the *emotional need* underneath the craving without using food\n\n"
            "The goal is never restriction. The goal is *understanding* — and then *freedom*.\n\n"
            "Is there a specific trigger or pattern you'd like to explore?"
        )

    # ── Cortisol / weight / biology
    if any(w in q for w in ['cortisol','weight gain','why can\'t i lose weight','diet not working','metabolism','leptin','ghrelin','hormone','biology','science','why diets fail']):
        return (
            "This is such an important question — and the science is on your side. 🌿\n\n"
            "When your body is under chronic stress, your **HPA axis** (hypothalamic-pituitary-adrenal axis) "
            "stays activated, keeping cortisol elevated. Here's what elevated cortisol does to your body:\n\n"
            "🔬 **Blocks leptin** — your satiety hormone, so you never feel full\n"
            "🔬 **Spikes ghrelin** — your hunger hormone, so you always feel hungry\n"
            "🔬 **Drives fat storage** — especially around the abdomen\n"
            "🔬 **Shuts down the prefrontal cortex** — making impulse control nearly impossible\n"
            "🔬 **Disrupts sleep** — which further worsens all of the above\n\n"
            "This means that if you try to diet while your nervous system is dysregulated, "
            "your *biology* is working against you — not your willpower.\n\n"
            "The Safe Body addresses the root: **regulate the nervous system first, and the body follows**.\n\n"
            "Would you like to learn more about the science, or how the program addresses this?"
        )

    # ── Sleep
    if any(w in q for w in ['sleep','can\'t sleep','insomnia','tired','exhausted','fatigue','rest']):
        return (
            "Sleep is not a luxury in The Safe Body — it is a *healing requirement*. 🌿\n\n"
            "Here's why: when you're sleep-deprived, cortisol rises, ghrelin spikes (hunger hormone), "
            "and leptin crashes (fullness hormone). Your body is biologically driven to eat more "
            "and store more fat — regardless of your intentions.\n\n"
            "In **Week 8 of The Safe Body**, Dr. Clarisse guides you through building your personal "
            "**Sleep Sanctuary Protocol** — a 5-step evening routine designed to:\n"
            "✦ Lower cortisol before bed\n"
            "✦ Signal safety to your nervous system\n"
            "✦ Restore your natural sleep-wake rhythm\n"
            "✦ Create a physical environment that supports deep rest\n\n"
            "Many women in the program report that their sleep improves dramatically by Week 8 — "
            "and their cravings and emotional eating reduce significantly as a result.\n\n"
            "Is sleep a particular challenge for you right now?"
        )

    # ── Movement
    if any(w in q for w in ['exercise','movement','workout','gym','joyful movement','hate exercise','can\'t exercise','physical activity']):
        return (
            "In The Safe Body, we do not talk about exercise as punishment. 🌿\n\n"
            "We talk about **joyful movement** — movement that your body *wants* to do, "
            "not movement you force it through out of shame or obligation.\n\n"
            "In **Week 7**, Dr. Clarisse guides you to:\n"
            "✦ Identify the difference between punishing movement and nourishing movement\n"
            "✦ Discover what types of movement genuinely bring you joy\n"
            "✦ Reconnect with your body as a partner — not an enemy to be disciplined\n"
            "✦ Build a movement practice that is *sustainable* because it feels good\n\n"
            "*\"Your body is not a problem to be solved. It is a home to be returned to.\"* — Dr. Clarisse\n\n"
            "Would you like to explore more about the program's approach to movement?"
        )

    # ── Faith
    if any(w in q for w in ['faith','god','spiritual','church','religion','christian','prayer','bible']):
        return (
            "Faith is honored in The Safe Body — always. 🌿\n\n"
            "Dr. Clarisse understands that for many women, their relationship with God and their "
            "spiritual community is central to who they are. The Safe Body creates space for that.\n\n"
            "The program does not require faith — but for women whose spirituality is a source of "
            "strength, Dr. Clarisse integrates faith-based perspectives into the healing work, including:\n"
            "✦ Reconciling body shame with spiritual identity\n"
            "✦ Using prayer and contemplative practices as regulation tools\n"
            "✦ Understanding your body as sacred — worthy of care, not punishment\n\n"
            "You do not have to choose between your faith and your healing. In The Safe Body, "
            "they walk together.\n\n"
            "Would you like to learn more about the program?"
        )

    # ── Boundaries / self-worth
    if any(w in q for w in ['boundary','boundaries','self worth','self-worth','people pleasing','people-pleasing','say no','deserve','worthy']):
        return (
            "Boundaries are one of the most powerful — and most overlooked — tools for sustainable weight loss. 🌿\n\n"
            "Here's what the research shows: chronic people-pleasing keeps your nervous system in a "
            "constant state of low-grade threat. Your body never fully relaxes. Cortisol stays elevated. "
            "Emotional eating continues.\n\n"
            "In **Week 9 of The Safe Body**, you'll explore:\n"
            "✦ Why saying yes to everyone else means saying no to your own healing\n"
            "✦ How to identify where your boundaries have been eroded — and why\n"
            "✦ Scripts for setting boundaries with compassion and confidence\n"
            "✦ How your self-worth is not determined by your weight, your productivity, or your service to others\n\n"
            "*You are worthy of healing right now — not after you lose the weight.*\n\n"
            "Is this something you're navigating in your own life?"
        )

    # ── General question with retrieved context
    if retrieved and retrieved[0]['score'] > 0.08:
        best = retrieved[0]
        # Extract a clean excerpt (first 300 chars)
        excerpt = best['text'][:350].rsplit(' ', 1)[0] + '…'
        topic   = best['topic']
        return (
            f"That's a beautiful question, and it touches on something we explore deeply in *{topic}*. 🌿\n\n"
            f"{excerpt}\n\n"
            "This is exactly the kind of understanding The Safe Body is built on — evidence-based, "
            "compassionate, and rooted in the truth that your body has always been doing its best to keep you safe.\n\n"
            "Would you like to go deeper on this topic, or is there something else on your heart?"
        )

    # ── Off-topic / out of scope
    return (
        "That's a thoughtful question — though it's a little outside what I'm designed to support here. 🌿\n\n"
        "I'm *Become Free*, and I'm here specifically to walk alongside you on topics related to "
        "The Safe Body program: nervous system healing, emotional eating, inner child work, "
        "mindful nourishment, joyful movement, and the path to sustainable freedom in your body.\n\n"
        "If you're looking for medical advice or diagnosis, please reach out to your personal healthcare provider.\n\n"
        "Is there something about your healing journey I can support you with today?"
    )

# ── API Endpoints ─────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str

@app.post("/chat")
async def chat(req: ChatRequest):
    query = req.message.strip()

    if not query:
        return JSONResponse({"reply": "I'm here and listening. What's on your heart today? 🌿"})

    # Crisis check — highest priority
    if is_crisis(query):
        return JSONResponse({
            "reply": (
                "I hear you, and I want you to know — you matter deeply. 💚\n\n"
                "What you're feeling right now is real, and you deserve real human support. "
                "Please reach out to the **988 Suicide & Crisis Lifeline** by calling or texting **988**. "
                "They are available 24/7 and they care.\n\n"
                "If you are in immediate danger, please call **911**.\n\n"
                "You are not alone."
            ),
            "is_crisis": True
        })

    # Retrieve relevant context
    retrieved = retrieve(query, top_k=5)

    # Build response
    reply = build_response(query, retrieved)

    return JSONResponse({
        "reply": reply,
        "topic": retrieved[0]['topic'] if retrieved else "General",
        "is_crisis": False
    })

@app.get("/health")
async def health():
    return {"status": "ok", "chunks": len(chunks)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
