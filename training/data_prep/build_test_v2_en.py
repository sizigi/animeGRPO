"""Build English test_v2 (50 sentences) with same group/subgroup/emotion structure
as the Japanese test_v2.parquet, for cross-lingual GRPO + AnimeScore evaluation.

Each entry preserves the perceptual role (high_arousal / low_arousal / kawaii /
character_archetype / neutral / homophone / loanword / formal / numeric / narrative)
but uses idiomatic English (not literal translation) so a native English listener
hears the same emotional/stylistic intent.
"""
import pandas as pd
from pathlib import Path

# Each tuple: (id, group, subgroup, emotion_label, text)
ROWS = [
    # G1A: high_arousal (4)
    ("S01", "G1_emotional", "high_arousal",  "excitement",
     "Yes! I finally got my hands on the new game!"),
    ("S02", "G1_emotional", "high_arousal",  "surprise",
     "Wait, seriously? I can't believe it, what a crazy coincidence!"),
    ("S03", "G1_emotional", "high_arousal",  "anger",
     "I can't take this anymore! Why won't anyone listen to me?!"),
    ("S04", "G1_emotional", "high_arousal",  "enthusiastic_announcement",
     "Everyone, come over here! You absolutely cannot miss today's event!"),
    # G1B: low_arousal (4)
    ("S05", "G1_emotional", "low_arousal",   "sadness",
     "Whenever I think back to that day, my heart still aches a little."),
    ("S06", "G1_emotional", "low_arousal",   "calm_reflection",
     "Drinking tea alone on a quiet night is one of my little joys."),
    ("S07", "G1_emotional", "low_arousal",   "gentle_explanation",
     "You see, that flower slowly starts to bloom when spring arrives."),
    ("S08", "G1_emotional", "low_arousal",   "quiet_disappointment",
     "I'm feeling a little tired, so I think I'll turn in early tonight."),
    # G1C: mixed (4)
    ("S09", "G1_emotional", "mixed",         "nostalgia",
     "I just suddenly remembered playing here with my friends a long time ago."),
    ("S10", "G1_emotional", "mixed",         "mild_curiosity",
     "Huh, I wonder if I've actually walked down this road before."),
    ("S11", "G1_emotional", "mixed",         "mild_satisfaction",
     "I actually managed to get up early today, and I feel pretty good somehow."),
    ("S12", "G1_emotional", "mixed",         "mild_concern",
     "They're saying tomorrow's weather might just turn into rain, you know."),
    # G2A: kawaii (5)
    ("S13", "G2_anime_stylized", "kawaii",   None,
     "Wooow, this candy is sooo cute! It feels like a shame to eat it!"),
    ("S14", "G2_anime_stylized", "kawaii",   None,
     "Hmph, you're being mean to me again! Fine, I don't care anymore!"),
    ("S15", "G2_anime_stylized", "kawaii",   None,
     "Right, right! I've been thinking exactly the same thing the whole time!"),
    ("S16", "G2_anime_stylized", "kawaii",   None,
     "Good morning! Let's play together today too, okay? It's a promise!"),
    ("S17", "G2_anime_stylized", "kawaii",   None,
     "Hehehe, it's a little secret, okay? You can't tell anyone, alright?"),
    # G2B: character_archetype (5)
    ("S18", "G2_anime_stylized", "character_archetype", None,
     "I'll never give up, no matter what! I'm gonna show you what I can really do!"),
    ("S19", "G2_anime_stylized", "character_archetype", None,
     "It's, it's not like I did it for you or anything! Don't get the wrong idea!"),
    ("S20", "G2_anime_stylized", "character_archetype", None,
     "You still know nothing of the true nature of this world."),
    ("S21", "G2_anime_stylized", "character_archetype", None,
     "Alright everyone, are you ready? We're going in at full power!"),
    ("S22", "G2_anime_stylized", "character_archetype", None,
     "It's okay, I'm right here beside you. There's nothing to worry about."),
    # G3: neutral conversational (8)
    ("S23", "G3_neutral", "conversational",  None,
     "I had lunch with my friend at a new café yesterday."),
    ("S24", "G3_neutral", "conversational",  None,
     "This weekend my whole family is planning to go to the park nearby."),
    ("S25", "G3_neutral", "conversational",  None,
     "You can easily look up the recipe for this dish online."),
    ("S26", "G3_neutral", "conversational",  None,
     "I heard the new building by the station is opening for business next month."),
    ("S27", "G3_neutral", "conversational",  None,
     "I check the news every morning while drinking my coffee."),
    ("S28", "G3_neutral", "conversational",  None,
     "If you transfer trains at the next station, you'll get there a bit faster."),
    ("S29", "G3_neutral", "conversational",  None,
     "I prefer autumn over summer because it's just easier to get through the day."),
    ("S30", "G3_neutral", "conversational",  None,
     "Lately I haven't been able to find time to read, and it's been bothering me."),
    # G1A extra: high_arousal (2)
    ("S31", "G1_emotional", "high_arousal",  "joyful_invitation",
     "Yes! Let's all go to the festival together this time, okay?"),
    ("S32", "G1_emotional", "high_arousal",  "intense_disbelief",
     "No way?! I can't believe something like this is actually happening!"),
    # G1B extra: low_arousal (2)
    ("S33", "G1_emotional", "low_arousal",   "gentle_reassurance",
     "It's okay, take your time. Just go at your own pace, alright?"),
    ("S34", "G1_emotional", "low_arousal",   "quiet_contemplation",
     "When I just stare out the window for a while, my mind starts to settle down."),
    # G1C extra: mixed (2)
    ("S35", "G1_emotional", "mixed",         "mild_anticipation",
     "I hope I'm really ready for tomorrow's presentation, I'm getting a little nervous."),
    ("S36", "G1_emotional", "mixed",         "subtle_gratitude",
     "Thank you for always looking out for me, it really means so much to me."),
    # G2A extra: kawaii (2)
    ("S37", "G2_anime_stylized", "kawaii",   None,
     "Eeek! Look look, this plushie is sooo fluffy! I just want to squeeze it!"),
    ("S38", "G2_anime_stylized", "kawaii",   None,
     "Sorry sorry, I'm a bit busy right now, but I promise I'll listen properly later!"),
    # G2B extra: character_archetype (2)
    ("S39", "G2_anime_stylized", "character_archetype", None,
     "Hmph, it's not like I want it or anything. You can have it, really."),
    ("S40", "G2_anime_stylized", "character_archetype", None,
     "I can see how hard you're trying. Please don't push yourself too much, okay?"),
    # G3 extra: conversational (2)
    ("S41", "G3_neutral", "conversational",  None,
     "Today's meeting ended a little earlier than scheduled, which was a real help."),
    ("S42", "G3_neutral", "conversational",  None,
     "It's been getting chilly in the mornings and evenings lately, autumn is here."),
    # G5: linguistically challenging (5)
    ("S43", "G5_linguistically_challenging", "homophone",   None,
     "Their friends were waiting there, but they're not sure exactly where to meet."),
    ("S44", "G5_linguistically_challenging", "loanword",    None,
     "The application's interface design genuinely has a sleek, sophisticated aesthetic."),
    ("S45", "G5_linguistically_challenging", "loanword",    None,
     "Could you reschedule the online meeting to accommodate our updated deadline?"),
    ("S46", "G5_linguistically_challenging", "formal_vocab", None,
     "Regarding next fiscal year's budget allocation, we must deliberate carefully with the relevant departments."),
    ("S47", "G5_linguistically_challenging", "numeric",     None,
     "Our appointment is scheduled for April fifteenth, twenty twenty-six, at three twenty-five P.M."),
    # G4: long_form_narrative (3)
    ("S48", "G4_long_form_narrative", "narrative", None,
     "When I visited the town where I grew up after such a long time, I found that the park where I used to play with my friends was still standing there unchanged, and somehow my heart just felt completely full."),
    ("S49", "G4_long_form_narrative", "narrative", None,
     "Cherry blossoms only bloom for such a brief moment each year, and that's exactly why we end up feeling like we have to cherish and savor every single second of that short window of time."),
    ("S50", "G4_long_form_narrative", "narrative", None,
     "You know that movie I watched yesterday? At first I really thought it was going to be pretty average, but then in the second half the plot suddenly got incredibly interesting, and by the end I was completely moved to tears."),
]

assert len(ROWS) == 50, f"expected 50 rows, got {len(ROWS)}"
assert {r[0] for r in ROWS} == {f"S{i:02d}" for i in range(1, 51)}, "id sequence mismatch"


def build_row(idx, sid, group, subgroup, emo, text):
    prompt = [
        {"content": f"Convert the text to speech:<|TEXT_UNDERSTANDING_START|>{text}<|TEXT_UNDERSTANDING_END|>",
         "role": "user"},
        {"content": "<|SPEECH_GENERATION_START|>", "role": "assistant"},
    ]
    return {
        "data_source":  "en_tts_mixed_v2",
        "prompt":       prompt,
        "ability":      "text-to-speech",
        "reward_model": {"ground_truth": text, "style": "rule"},
        "extra_info":   {
            "emotion_label": emo,
            "group": group,
            "id": sid,
            "index": idx,
            "split": "test",
            "subgroup": subgroup,
            "text": text,
            "validation_purpose": "cross-lingual AnimeScore+CER GRPO evaluation",
        },
    }


def main():
    out_dir = Path("~/data/llasa-tts-rl-grpo-en-v2").expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [build_row(i, *r) for i, r in enumerate(ROWS)]
    df = pd.DataFrame(rows)

    out = out_dir / "test_v2.parquet"
    df.to_parquet(out, index=False)
    print(f"wrote {out} with {len(df)} rows")

    # Sanity print
    from collections import Counter
    grp = Counter(r["extra_info"]["group"] for _, r in df.iterrows())
    sub = Counter(r["extra_info"]["subgroup"] for _, r in df.iterrows())
    print(f"group dist: {dict(grp)}")
    print(f"subgroup dist: {dict(sub)}")


if __name__ == "__main__":
    main()
