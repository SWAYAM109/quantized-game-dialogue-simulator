"""Expands data/game_dialogue.txt with templated RPG dialogue (seeded, reproducible)."""
import random
from pathlib import Path

random.seed(42)
PATH = Path(__file__).resolve().parents[1] / "data" / "game_dialogue.txt"
places = ["the old observatory", "the river of ash", "the northern gate", "the whispering forest", "the sunken temple", "the iron bridge", "the frozen pass", "the ruined citadel"]
items = ["a healing potion", "an ancient map", "a silver key", "the crystal shard", "a torn banner", "a rusted blade", "a lantern of embers"]
foes = ["the shadow army", "three riders", "the ash wolves", "the hollow knights", "a dragon", "the bandit king"]
npcs = ["Elder", "Scout", "Merchant", "Companion", "Blacksmith", "Guard"]
tmpl_npc = [
    "Be careful near {p}. {f} was seen there at dusk.",
    "I can offer you {i} for the road. Spend it wisely.",
    "The villagers say {f} will strike before the next moon.",
    "If you reach {p}, you will find {i} waiting.",
    "Few return from {p}, but you are not like the others.",
    "Take {i}. You will need it when {f} arrives.",
    "The path to {p} is long. Rest here tonight.",
]
tmpl_hero = [
    "We leave at dawn. Show me the way to {p}.",
    "I will find {i} and end this before {f} reaches the village.",
    "Stay close. If {f} appears, run for {p}.",
    "Thank you. I will not waste {i}.",
    "Then there is no time to lose. Lead on.",
    "I have fought {f} before. I am not afraid.",
]
lines = []
for _ in range(420):
    npc = random.choice(npcs)
    ctx = dict(p=random.choice(places), i=random.choice(items), f=random.choice(foes))
    lines.append(f"{npc}: {random.choice(tmpl_npc).format(**ctx)}")
    lines.append(f"Hero: {random.choice(tmpl_hero).format(**ctx)}")
base = [l for l in PATH.read_text(encoding="utf-8").splitlines() if l.strip() and not l.startswith("#")]
PATH.write_text("\n".join(base[:40] + lines) + "\n", encoding="utf-8")
print(f"Wrote {len(base[:40]) + len(lines)} lines to {PATH}")
