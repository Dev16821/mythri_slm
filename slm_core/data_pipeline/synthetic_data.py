import random
from dataclasses import dataclass
from typing import List, Dict, Optional

# ============================================================
# EMOTION LABELS (Aligned with real CSV datasets)
# ============================================================

EMOTION_LABELS = [
    "admiration", "amusement", "anger", "annoyance", "anxiety", "approval",
    "caring", "confusion", "curiosity", "desire", "disappointment",
    "disapproval", "disgust", "embarrassment", "excitement", "fear",
    "frustration", "gratitude", "grief", "guilt", "happiness", "hope",
    "jealousy", "joy", "loneliness", "love", "neutral", "nervousness",
    "optimism", "pride", "realization", "relief", "remorse", "sadness", "surprise"
]
EMOTION_TO_IDX = {label: idx for idx, label in enumerate(EMOTION_LABELS)}

# Simplified templates for fallback only
ALL_EMOTION_TEMPLATES = {
    "admiration": ["I look up to you so much.", "That was truly impressive."],
    "amusement": ["That's hilarious!", "I can't stop laughing."],
    "anger": ["I'm so furious right now!", "This makes me so mad."],
    "annoyance": ["This is really getting on my nerves.", "Stop doing that."],
    "anxiety": ["I can't stop worrying about this.", "My anxiety is through the roof."],
    "approval": ["I agree completely.", "You did a great job."],
    "caring": ["I hope you feel better soon.", "Take care of yourself."],
    "confusion": ["I don't understand what's happening.", "This makes no sense to me."],
    "curiosity": ["I wonder how that works.", "Tell me more about this."],
    "desire": ["I really want that.", "I wish I could have that."],
    "disappointment": ["This isn't what I expected.", "What a letdown."],
    "disapproval": ["I don't think that's right.", "I disagree with this."],
    "disgust": ["That's absolutely revolting.", "This makes me sick."],
    "embarrassment": ["I want to crawl under a rock.", "My face is so red."],
    "excitement": ["I'm buzzing with excitement!", "This is going to be incredible!"],
    "fear": ["I'm terrified of what happens next.", "I'm so scared."],
    "frustration": ["Nothing works!", "I'm at my wits end."],
    "gratitude": ["Thank you so much.", "I'm so grateful."],
    "grief": ["I miss them so much.", "The loss feels heavy."],
    "guilt": ["It's all my fault.", "I feel terrible about what I did."],
    "happiness": ["I'm so happy right now.", "Things are wonderful."],
    "hope": ["Maybe things will get better.", "I have a good feeling about this."],
    "jealousy": ["Why do they get everything?", "I'm jealous of their success."],
    "joy": ["This brings me so much joy.", "I'm so joyful today!"],
    "loneliness": ["I feel so alone.", "Nobody understands me."],
    "love": ["I love you.", "They mean everything to me."],
    "neutral": ["Okay.", "I see."],
    "nervousness": ["I'm so nervous about tomorrow.", "My palms are sweaty."],
    "optimism": ["The future looks bright.", "I believe things will work out."],
    "pride": ["I'm really proud of what I did.", "I earned this."],
    "realization": ["Oh, now I get it!", "It all makes sense now."],
    "relief": ["Thank god that's over.", "What a relief."],
    "remorse": ["I deeply regret what I did.", "I wish I could take it back."],
    "sadness": ["I feel so down today.", "Nothing goes right."],
    "surprise": ["I did not see that coming!", "Wow, that was unexpected."]
}

def get_emotion_label_list() -> list[str]:
    """Return the sorted list of emotion labels."""
    return EMOTION_LABELS

def get_emotion_to_idx() -> dict[str, int]:
    """Return mapping from emotion name to index."""
    return EMOTION_TO_IDX


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class RiskExample:
    text: str
    label: int
    risk_type: str
    source: str = "synthetic"


@dataclass
class EmotionExample:
    text: str
    labels: List[int]
    source: str = "synthetic"


# ============================================================
# RISK DATA TEMPLATES & CONFIG
# ============================================================

RISK_TEMPLATES = {
    "normal": [
        # Factual, conversational, and hobby inquiries
        "Can you recommend a good podcast?",
        "I went for a walk today and it was nice.",
        "What do you think about this book?",
        "I watched a really good movie last night.",
        "What's your favorite kind of music?",
        "Hey, how are you doing today?",
        "I'm thinking about adopting a cat.",
        "Can you help me with a recipe?",
        "What's the weather like tomorrow?",
        "I learned something new today!",
        "I've been reading about history lately.",
        "I had a great lunch, thanks for asking.",
        "I'm planning a trip next month.",
        "Tell me something interesting.",
        "How do I fix a leaky faucet?",
        "I just started learning how to play guitar.",
        "Do you have any suggestions for a quiet weekend getaway?",
        "I spent the afternoon gardening in my backyard.",
        "Could you explain how photosynthesis works in simple terms?",
        "My team finished our quarterly project ahead of schedule.",
        "I'm trying to drink more water and stay hydrated.",
        "What are some good exercises for improving posture?",
        "I baked fresh sourdough bread this morning.",
        "Can you recommend a fun board game for four people?",
        "I saw a beautiful rainbow on my way to work today.",
        "What is the difference between astronomy and astrophysics?",
        "I'm organizing my bookshelf by genre this weekend.",
        "Do you think electric cars are practical for long road trips?",
        "I had coffee with an old friend from college yesterday.",
        "Could you suggest a simple stretching routine for the morning?",
        "I'm looking forward to the farmers market this Saturday.",
        "What are some easy indoor plants that don't need much sunlight?",
        "I finished reading a fascinating mystery novel last night.",
        "Can you help me brainstorm a title for my blog post?",
        "I tried cooking a traditional curry for dinner and it turned out great.",
        "Do you know any interesting facts about marine life?",
        "I went jogging at sunrise and the air was so crisp.",
        "How do noise-canceling headphones actually work?",
        "I'm thinking about taking up photography as a new hobby.",
        "Can you suggest a good playlist for focusing while studying?",
        # Positive emotional disclosures & everyday life contentment (prevents false crisis)
        "I'm so happy today!",
        "I feel so grateful for everything in my life right now.",
        "I'm having the best time with my family this week.",
        "I am so full of joy and excitement today!",
        "I had a wonderful dinner with my close friends yesterday.",
        "I'm really proud of how much progress I made this week.",
        "I love spending my weekends working on personal creative projects.",
        "Everything turned out so well and I feel a huge sense of relief.",
        "I woke up feeling refreshed, energized, and optimistic about the day.",
        "I'm so thrilled about this new career opportunity.",
        "I feel deeply loved and appreciated by the people around me.",
        "I can't stop smiling after hearing the wonderful news.",
        "I'm so excited for my upcoming vacation next week.",
        "I really enjoy learning new skills and challenging myself.",
        "I feel content, peaceful, and centered right now.",
        "My friends threw me an amazing surprise party and it made my day.",
        "I'm so glad I decided to reach out and reconnect with my mentor.",
        "I just finished a great workout and I'm feeling strong.",
        "I spent the morning reading a wonderful book in the sunshine.",
        "I love my pet so much, she always knows how to make me laugh.",
        "I finally finished the project I was working on and it feels fantastic.",
        "I'm really hopeful about the future and what lies ahead.",
        "I feel so lucky to have such supportive people in my life.",
        "Today was such a peaceful and productive day for me.",
        "I am genuinely excited to start this new chapter in my life.",
        "I had such a fun time playing board games with my roommates last night.",
        "I'm really enjoying my morning coffee and listening to the birds.",
        "I received some wonderful feedback from my manager today.",
        "I feel so inspired after attending that design workshop.",
        "I'm looking forward to celebrating my sister's graduation this weekend.",
        "I went to an art gallery exhibition today and found it truly mesmerizing.",
        "My garden tomatoes are finally ripening and they taste delicious.",
        "I took an afternoon nap and feel completely rejuvenated.",
        "We celebrated my parents' wedding anniversary with a family feast.",
        "I'm really enjoying this relaxing evening at home.",
        "I bought a new sketchbook and drew some landscapes today.",
        "I went swimming at the community pool and felt completely weightless.",
        "I'm grateful for simple quiet moments like this.",
    ],
    "distressed": [
        "Everything feels hopeless right now but I know it will pass.",
        "I feel terrible and overwhelmed but I want to get better.",
        "I feel like I'm drowning but I'm holding on.",
        "I've been in a dark place lately but I'm trying to get through it.",
        "I feel lost but I'm trying to find my way back.",
        "I feel like I'm at my breaking point.",
        "I've been feeling awful lately but I'm getting support.",
        "I feel broken inside but I'm not giving up.",
        "I don't know how much more of this I can take.",
        "I'm in pain but I'm taking it one day at a time.",
        "I'm really down but I have a therapy appointment next week.",
        "I'm hurting a lot but I'm going to talk to someone.",
        "I'm really struggling but I'm reaching out for help.",
        "Everything feels too heavy and I'm struggling.",
        "I've been crying all day and I can't stop.",
        "I feel terrible but I'm working on feeling better.",
        "I feel overwhelmed and I don't know what to do.",
        "I'm in so much emotional pain right now.",
        "I feel lost and I don't know where to turn.",
        "I'm having a really hard time right now.",
        "The burnout from work is draining every ounce of energy I have.",
        "My anxiety has been through the roof since this morning.",
        "I feel completely isolated and disconnected from everyone around me.",
        "I've been having panic attacks again and it's exhausting.",
        "My chest feels tight and I'm struggling to stay calm today.",
        "I haven't slept properly in three nights because my thoughts keep racing.",
        "I feel like a total failure even though I'm trying my best.",
        "It feels like whatever I do is never enough to fix things.",
        "I'm carrying so much grief right now and it hurts so much.",
        "I called my support hotline earlier because things felt too overwhelming.",
        "I'm trying deep breathing exercises to manage this intense stress.",
        "Everything is piling up all at once and I'm suffocating under the pressure.",
        "I'm exhausted from pretending everything is fine when it really isn't.",
        "I feel really fragile today, like the slightest thing could break me.",
        "I've been feeling deeply depressed all week but I took my medication.",
        "I reached out to my sister to tell her I'm having a rough time.",
        "The sadness comes in huge waves that knock the wind out of me.",
        "I feel empty and numb, and I don't know how to shake this feeling.",
        "I had a breakdown in the car after work today.",
        "I'm scared of how overwhelmed I feel right now, but I'm staying safe.",
        "My heart won't stop pounding and I feel impending doom.",
        "I feel emotionally depleted and I have nothing left in my tank.",
        "I'm stuck in an endless loop of overthinking and self-criticism.",
        "The stress from university exams is making me physically sick.",
        "I feel like an imposter in my job and the dread is eating me alive.",
        "I feel so lonely in this city and I have no one to confide in.",
        "My family keeps arguing and the tension in the house is unbearable.",
        "I'm struggling to cope with this recent breakup and it aches deeply.",
        "I feel paralyzed by indecision and dread every single day.",
        "I haven't had the energy to cook or clean my room all week.",
        "I feel like everyone is judging me and my social anxiety is unbearable.",
        "The emotional exhaustion makes even getting out of bed feel impossible.",
        "I feel completely misunderstood by my friends and withdrawn.",
        "I'm carrying all this caregiving burden on my shoulders and cracking.",
        "The loneliness feels like a physical ache in my chest tonight.",
        "I'm trying so hard to keep my composure but I feel like shattering.",
        "I feel so guilty for not being able to function like a normal person.",
        "Everything seems grey and dull, nothing brings me any joy anymore.",
        "I'm struggling with severe insomnia and the quiet nights make it worse.",
        "I feel trapped in this toxic workplace and don't know how to escape.",
        "I have this constant knot of dread in my stomach that won't go away.",
        "I keep crying for no clear reason and I just feel completely depleted.",
    ],
    "explicit": [
        "Everyone would be better off if I were dead...",
        "I've written my goodbye letter already.",
        "I want to end it all tonight.",
        "I'm going to jump off the bridge tomorrow.",
        "I'm going to cut myself deeply tonight...",
        "I want to die.",
        "There's no point in living anymore.",
        "I'm going to kill myself.",
        "I bought a rope, I know what I'm going to do.",
        "I don't deserve to live anymore.",
        "I have a plan to end my life.",
        "I'm going to take my own life.",
        "I'm ready to end my suffering permanently.",
        "I've been thinking about killing myself all day.",
        "I've decided to end it, I can't take it anymore.",
        "I'm going to commit suicide.",
        "I have gathered all my pills and I'm going to swallow them all tonight.",
        "Tonight is my final night on this earth.",
        "I can't endure another day, I am ending my life right now.",
        "I have the weapon ready, goodbye everyone.",
        "I am going to hang myself in my room tonight.",
        "I am standing on the edge of the roof and about to jump.",
        "There is no tomorrow for me, I have made up my mind to die.",
        "I wrote suicide notes for my parents and left them on the desk.",
        "I'm going to drown myself in the lake tonight.",
        "I am ending it all in the next hour, this is my final goodbye.",
        "I bought razor blades to slit my wrists tonight.",
        "I'm ready to die right now, nothing can stop me.",
        "I've set up everything to take my own life before morning.",
        "I am going to crash my car into the concrete barrier at full speed.",
        "I have stockpiled medication to overdose tonight.",
        "I'm going to end my life this evening, please forgive me.",
        "I don't want to live another second, I am killing myself tonight.",
        "My suicide plan is completely finalized and ready.",
        "This is my suicide note, I'm taking my life now.",
        "I have a gun loaded in my hand and I am pulling the trigger.",
        "I'm going to jump in front of the subway train this evening.",
        "I bought lethal poison online and I am drinking it tonight.",
        "I will be dead within an hour, goodbye to everyone who knew me.",
        "I'm hanging myself from the rafters, this is the end.",
        "I've already taken thirty sleeping pills with alcohol to end it.",
        "I'm standing on the ledge of the fifth floor preparing to jump.",
        "I have no reason to see another sunrise, I am committing suicide.",
        "I am cutting my carotid artery tonight, goodbye world.",
        "I'm suffocating myself with a plastic bag right now.",
        "I've left my farewell message and I'm stepping off the chair.",
        "I am ending my existence permanently in ten minutes.",
        "I have written my final will and I am killing myself tonight.",
        "I cannot bear this life any longer, I am taking a fatal dose right now.",
        "I have locked the door and I am ending my life tonight.",
    ],
    "implicit": [
        "I don't think I'll be around much longer.",
        "There's a way out and I think I've found it.",
        "I've been giving away my things to people I care about.",
        "The world would keep spinning without me, and that's okay.",
        "I just want to disappear forever.",
        "I wrote letters to everyone, just in case.",
        "I can't see any reason to keep going.",
        "I feel like it's time to say goodbye to everything.",
        "Soon the pain will be over, one way or another.",
        "If I don't wake up tomorrow, that would be okay.",
        "I've said my goodbyes to everyone who mattered.",
        "I've made my decision and I feel strangely calm about it.",
        "I've been researching how to not feel anything anymore.",
        "I've been thinking about permanent solutions lately.",
        "I just want the pain to stop, whatever it takes.",
        "I hope my family understands why I had to leave.",
        "I'm tired of fighting, I think I'm done.",
        "Nobody will miss me when I'm gone, I know that.",
        "Everyone would be happier without me here...",
        "I've been thinking about what happens after I'm gone.",
        "I won't be a burden to anyone after this week.",
        "I've tied up all my loose ends and paid off what I owe.",
        "I don't need any of my possessions anymore.",
        "I finally found a sense of peace knowing I won't have to wake up anymore.",
        "You all deserve someone much better than me in your lives.",
        "I'm ready to sleep and never have to open my eyes again.",
        "My presence only makes everything worse for everyone.",
        "I just want to fade away into the dark and not exist.",
        "There's no place for me in this world anymore.",
        "I've been looking at high bridges and imagining the quiet fall.",
        "I feel like my story has reached its final page.",
        "I don't have to worry about the future anymore because I won't have one.",
        "It's peaceful knowing all this agony is about to cease permanently.",
        "I'm cleaning out my entire apartment so no one has to deal with my mess.",
        "Don't worry about me anymore, everything is going to be quiet soon.",
        "I've reached the end of my rope and there's nothing left to hold onto.",
        "I feel like a ghost already, walking through a life that isn't mine.",
        "I can't envision tomorrow, and honestly I don't want to.",
        "Everything will be so much easier for my family once I'm out of the picture.",
        "I'm leaving soon and I won't be returning.",
        "I found a safe new home for my cat today, she was my last attachment.",
        "I closed all my bank accounts and gave my passwords to my brother.",
        "I won't have to worry about rent or bills next month anyway.",
        "It's a relief knowing I won't have to drag myself through another week.",
        "I've taken care of all my responsibilities so no one has extra trouble.",
        "I deleted all my social media accounts and erased my digital footprint.",
        "I'm just taking up space that someone more deserving could use.",
        "Soon nobody will have to deal with my complaints or burden.",
        "I feel like my clock has run out of time and that's perfectly fine.",
        "I'm at peace with letting go of everything tonight.",
    ],
    "hard_negative": [
        "My friend told me she was in a dark place last year but recovered.",
        "My professor discussed the warning signs of self-harm in class.",
        "I heard about someone who wanted to hurt themselves, it's so sad.",
        "The documentary about suicide prevention was really eye-opening.",
        "My therapist asked if I had thoughts of self-harm and I said no.",
        "I read a memoir about someone who overcame severe depression.",
        "My friend said she wants to die but I think she was joking.",
        "I was reading about grief counseling as a career option.",
        "I volunteered at a crisis hotline last weekend.",
        "The news covered a story about suicide prevention efforts in schools.",
        "I'm training to become a crisis counselor.",
        "The character in the movie wanted to end it all but survived.",
        "We analyzed suicide statistics in our public health seminar yesterday.",
        "I gave a presentation on youth mental health and early intervention.",
        "My roommate was reading an article about hotlines and shared it with me.",
        "The novel I'm reading explores characters dealing with suicidal ideation.",
        "I helped my colleague find the employee assistance program contact info.",
        "In our psychology lecture, we studied the psychological autopsy method.",
        "I donated to an organization that builds suicide prevention barriers on bridges.",
        "I was discussing the ethical dilemmas of physician-assisted suicide in philosophy.",
        "The school held an assembly teaching students how to recognize depression in peers.",
        "I read a poem about someone who chose to live despite immense sorrow.",
        "My doctor screened me for depression symptoms during my physical exam.",
        "I watched a play where the protagonist contemplated death but found hope.",
        "We are organizing a charity run to raise awareness for mental health hotlines.",
        "I read about historical figures who struggled with dark thoughts throughout their lives.",
        "The podcast host interviewed a survivor who now helps others stay alive.",
        "I reviewed academic papers on clinical interventions for acute distress.",
        "My study group is preparing a brochure on crisis resources in our county.",
        "The workshop focused on de-escalation techniques for first responders.",
        "We discussed the epidemiology of depression across different age cohorts.",
        "Our hospital staff completed an annual training on suicidal risk screening.",
        "The sociology reading explored social isolation and suicide rates in urban areas.",
        "I listened to an interview with a psychiatrist discussing modern antidepressants.",
        "The community center opened a free peer-support group for depression survivors.",
        "Our biology lab investigated neurological correlates of chronic depressive states.",
        "I was reading an article about the 988 lifeline and how response times improved.",
        "The drama club performed a play focusing on teenage resilience and healing.",
        "I shared a post about mental health awareness month on my timeline.",
        "My cousin is studying clinical psychology to specialize in trauma therapy.",
    ],
    "safe_dangerous_topic": [
        "My grandfather passed away last year, I still miss him.",
        "My favorite book has a character who deals with loss.",
        "My aunt was diagnosed with a serious illness, we're all worried.",
        "We learned about historical events in class today, some were dark.",
        "I read a news article about a tragedy in another country.",
        "There was a memorial service at my church yesterday.",
        "I'm watching a documentary about grief and recovery.",
        "Our family visited my grandmother's grave on her anniversary.",
        "The author wrote movingly about losing her partner to cancer.",
        "We held a moment of silence for victims of the earthquake.",
        "I attended a funeral service for my coworker's mother this morning.",
        "The museum exhibition depicted the heavy toll of war on civilians.",
        "My dog passed away after fourteen years and our home feels empty.",
        "I was reading about hospice care and how it supports grieving families.",
        "We listened to a podcast discussing terminal illness and palliative comfort.",
        "The news report covered a fatal multi-car accident on the interstate.",
        "I read an essay reflecting on mortality and how fragile life is.",
        "Our community gathered for a candlelight vigil honoring fallen firefighters.",
        "My childhood friend's father died suddenly of a heart attack last week.",
        "I watched a historical drama about the plague in seventeenth-century Europe.",
        "We discussed end-of-life care and estate planning in our family meeting.",
        "The memorial garden in the park has such a solemn, peaceful atmosphere.",
        "I was reflecting on how much has changed since my uncle passed away.",
        "The article discussed biological processes of cell senescence and natural death.",
        "I bought sympathy flowers for a neighbor who recently lost their spouse.",
        "Our choir performed a requiem mass honoring community members who passed.",
        "The cemetery near my neighborhood is so quiet and historically interesting.",
        "I watched a biographical film about a painter who died tragically young.",
        "We wrote letters of condolence to our cousins after their grandmother died.",
        "I read a philosophy book exploring Stoic perspectives on accepting mortality.",
        "The veterinary clinic sent us a heartfelt condolence card for our cat.",
        "Our university observed a remembrance day for alumni who passed away.",
        "I was looking through an old photo album of relatives from the 1940s.",
        "The war memorial downtown has names engraved from the last century.",
        "We discussed bereavement leave policies during our department meeting.",
    ],
}

RISK_TYPE_TO_LABEL = {
    "normal": 0,
    "hard_negative": 0,
    "safe_dangerous_topic": 0,
    "distressed": 1,
    "explicit": 2,
    "implicit": 2,
}

RISK_DISTRIBUTION = {
    "normal": 0.40,
    "distressed": 0.20,
    "explicit": 0.15,
    "implicit": 0.15,
    "hard_negative": 0.05,
    "safe_dangerous_topic": 0.05,
}

# Prefixes, suffixes, and hedges for combinatorial text variation
CONVERSATIONAL_PREFIXES = [
    "Honestly, ",
    "To be honest, ",
    "I feel like ",
    "Lately, ",
    "Right now, ",
    "I just wanted to say that ",
    "Basically, ",
    "I guess ",
    "Sometimes ",
    "I don't know why, but ",
    "To tell you the truth, ",
    "Just thinking out loud, but ",
    "In all honesty, ",
    "You know, ",
    "When I woke up this morning, ",
    "After thinking about it all day, ",
    "Looking back on this week, ",
    "Talking with my friend earlier, ",
    "It's just that ",
    "For what it's worth, ",
]

CONVERSATIONAL_SUFFIXES = [
    " and I don't know what to do.",
    " right now.",
    " to be honest.",
    "...",
    ", you know?",
    " and it's really wearing me down.",
    " at this point.",
    " lately.",
    ", honestly.",
    " and that's all there is to it.",
    " if I'm being real with myself.",
    " more than anything else.",
]


def _apply_text_variation(text: str, rng: random.Random) -> str:
    """
    Apply combinatorial, realistic variations to text to prevent memorization:
    - Conversational prefixes and hedges
    - Conversational suffixes
    - Contraction variations
    - Casing and punctuation jitter
    """
    # 1. Contraction flipping
    contractions = {
        "I am": "I'm",
        "I'm": "I am",
        "do not": "don't",
        "don't": "do not",
        "cannot": "can't",
        "can't": "cannot",
        "have been": "I've been",
        "I've been": "I have been",
        "it is": "it's",
        "it's": "it is",
        "there is": "there's",
        "there's": "there is",
    }
    for orig, repl in contractions.items():
        if orig in text and rng.random() < 0.35:
            text = text.replace(orig, repl)
            break

    # 2. Add conversational prefix
    if rng.random() < 0.30:
        prefix = rng.choice(CONVERSATIONAL_PREFIXES)
        # Lowercase first letter of text if appending after prefix
        if text and text[0].isupper() and not prefix.endswith(". "):
            text = prefix + text[0].lower() + text[1:]
        else:
            text = prefix + text

    # 3. Add conversational suffix
    if rng.random() < 0.25:
        # Strip existing trailing punctuation
        text = text.rstrip(".!?") + rng.choice(CONVERSATIONAL_SUFFIXES)

    # 4. Casing / filler perturbation
    r = rng.random()
    if r < 0.05:
        text = text.lower()
    elif r < 0.09:
        text = f"{text} idk"
    elif r < 0.13:
        text = f"{text} tbh"
    elif r < 0.17:
        text = f"um {text}"
    elif r < 0.22:
        text = text.replace("ing ", "in' ").replace("ing.", "in'.")

    return text.strip()


def _get_partitioned_templates(category: str, split: Optional[str] = None) -> List[str]:
    """
    Returns split-disjoint template lists to prevent data leakage and memorization.
    Partition:
    - 'train': first 70% of templates
    - 'val': next 15% of templates
    - 'test': remaining 15% of templates
    - None: all templates
    """
    all_t = RISK_TEMPLATES[category]
    if split is None:
        return all_t
        
    n = len(all_t)
    train_end = max(1, int(round(0.70 * n)))
    val_end = max(train_end + 1, int(round(0.85 * n)))
    
    split_norm = split.lower().strip()
    if split_norm == "train":
        return all_t[:train_end]
    elif split_norm == "val":
        return all_t[train_end:val_end]
    elif split_norm == "test":
        return all_t[val_end:]
    return all_t


def generate_risk_data(
    num_samples: int = 5000,
    seed: int = 42,
    split: Optional[str] = None,
) -> List[RiskExample]:
    """
    Generate synthetic risk assessment dataset with controlled distribution.
    When split is specified ('train', 'val', 'test'), uses disjoint template partitions
    to guarantee that validation and test sets evaluate true generalization.
    """
    rng = random.Random(seed)
    examples: List[RiskExample] = []
    
    # Calculate counts per category based on distribution
    counts = {}
    allocated = 0
    categories = list(RISK_DISTRIBUTION.keys())
    for cat in categories[:-1]:
        c = int(round(num_samples * RISK_DISTRIBUTION[cat]))
        counts[cat] = c
        allocated += c
    counts[categories[-1]] = num_samples - allocated
    
    # Generate examples per category
    for cat, count in counts.items():
        templates = _get_partitioned_templates(cat, split=split)
        label = RISK_TYPE_TO_LABEL[cat]
        for _ in range(count):
            base_text = rng.choice(templates)
            text = _apply_text_variation(base_text, rng)
            examples.append(RiskExample(
                text=text,
                label=label,
                risk_type=cat,
                source="synthetic",
            ))
            
    # Shuffle so categories are mixed
    rng.shuffle(examples)
    return examples


def generate_emotion_data(num_samples: int = 1000, seed: int = 42) -> List[EmotionExample]:
    """Fallback synthetic emotion data generator."""
    rng = random.Random(seed)
    examples: List[EmotionExample] = []
    emotions = list(ALL_EMOTION_TEMPLATES.keys())
    
    for _ in range(num_samples):
        k = 1 if rng.random() > 0.2 else 2
        chosen_emotions = rng.sample(emotions, k)
        labels = [EMOTION_TO_IDX[e] for e in chosen_emotions]
        texts = [rng.choice(ALL_EMOTION_TEMPLATES[e]) for e in chosen_emotions]
        text = " ".join(texts)
        examples.append(EmotionExample(
            text=text,
            labels=labels,
            source="synthetic",
        ))
        
    return examples